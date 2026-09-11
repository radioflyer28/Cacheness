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
from uuid import uuid4

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
    AuthorityPublicationState,
    AuthorityIdentitySnapshot,
    AuthorityInventoryEntry,
    CandidateBatchReceipt,
    CandidateEntryReceipt,
    FinalizeReceipt,
    MigrationAuthority,
    RollbackReceipt,
    VerifiedCandidateReceipt,
    candidate_digest,
)
from .migration_evidence import (
    MaintenanceEvidenceState,
    MaintenanceEvidenceStore,
    MaintenanceRunEvidence,
    RebuildReceiptBatch,
    StoppedWorkerAcknowledgement,
    decode_bounded_canonical_json,
)
from .projections import (
    ProjectionController,
    ProjectionOutcome,
    ProjectionResult,
    ProjectionStatus,
)
from .read_contract import BlobReceipt


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MAX_COMPATIBILITY_TEXT_BYTES = 256
_MAX_PLAN_BYTES = 1_048_576
_MAX_PLAN_TEXT_BYTES = 262_144
_MAX_PLAN_DEPTH = 16
_MAX_PLAN_NODES = 16_384
_MAX_PLAN_COLLECTION_ITEMS = 4_096
_DEFAULT_MAX_ENTRIES_PER_RUN = 256
_DEFAULT_MAX_BYTES_PER_RUN = 64 * 1024 * 1024
_DEFAULT_MAX_EVIDENCE_BYTES_PER_RUN = 64 * 1024


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
    SOURCE_MISMATCH = "source_mismatch"
    DESTINATION_MISMATCH = "destination_mismatch"
    NON_EXECUTABLE_TRANSFORMATION = "non_executable_transformation"
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

    def supports(
        self,
        source: StoreVersionDimensions,
        destination: StoreVersionDimensions,
    ) -> bool:
        """Return whether both inspected endpoints exactly match this edge.

        The destination is an independently persisted contract.  A matching
        source alone can never authorize a copy or a native transformation.
        """
        return source == self.source and destination == self.destination


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


@dataclass(frozen=True)
class RebuildExclusion:
    """One explicit, bounded omission selected from a complete rebuild inventory.

    Rebuild starts with the entire inventory included.  An operator may name
    exact keys or one closed disposition category, but the resulting canonical
    plan always stores the resolved keys and a stable reason.  Wildcards and
    open-ended predicates therefore cannot change the set after confirmation.
    """

    kind: str
    reason: str
    keys: tuple[str, ...] = ()
    disposition: MigrationDisposition | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"keys", "disposition"}:
            raise ValueError("rebuild exclusion kind is invalid")
        object.__setattr__(self, "reason", _compatibility_text(self.reason, "exclusion reason"))
        if self.kind == "keys":
            if self.disposition is not None or not self.keys:
                raise ValueError("key exclusions require exact keys only")
            normalized = tuple(
                _compatibility_text(key, "excluded key") for key in self.keys
            )
            if len(normalized) != len(set(normalized)):
                raise ValueError("rebuild exclusion keys must be unique")
            object.__setattr__(self, "keys", tuple(sorted(normalized)))
            return
        if self.keys or not isinstance(self.disposition, MigrationDisposition):
            raise ValueError("category exclusions require one disposition")

    @classmethod
    def exact_keys(cls, keys: tuple[str, ...], *, reason: str) -> "RebuildExclusion":
        """Name one explicit immutable set of source keys for omission."""
        return cls(kind="keys", keys=keys, reason=reason)

    @classmethod
    def by_disposition(
        cls, disposition: MigrationDisposition, *, reason: str
    ) -> "RebuildExclusion":
        """Name one closed inventory category, resolved before confirmation."""
        return cls(kind="disposition", disposition=disposition, reason=reason)

    def resolve(
        self, entries: tuple[MigrationEntryAssessment, ...]
    ) -> Mapping[str, object]:
        """Resolve this selector against one fixed complete inventory."""
        entries_by_key = {entry.entry.key: entry for entry in entries}
        if self.kind == "keys":
            selected = self.keys
            unknown = set(selected) - set(entries_by_key)
            if unknown:
                raise ValueError("rebuild exclusion names a key outside the inventory")
            return MappingProxyType(
                {"kind": "keys", "keys": selected, "reason": self.reason}
            )
        assert self.disposition is not None
        selected = tuple(
            sorted(
                entry.entry.key
                for entry in entries
                if entry.disposition is self.disposition
            )
        )
        if not selected:
            raise ValueError("rebuild exclusion category matches no inventory entries")
        return MappingProxyType(
            {
                "disposition": self.disposition.value,
                "keys": selected,
                "kind": "disposition",
                "reason": self.reason,
            }
        )


def _normalize_rebuild_exclusions(
    exclusions: tuple[Mapping[str, object], ...],
    entries: tuple[MigrationEntryAssessment, ...],
) -> tuple[Mapping[str, object], ...]:
    """Validate canonical rebuild omissions against the complete plan inventory."""
    inventory = {entry.entry.key: entry for entry in entries}
    selected_keys: set[str] = set()
    normalized: list[Mapping[str, object]] = []
    for exclusion in exclusions:
        if not isinstance(exclusion, Mapping):
            raise TypeError("rebuild exclusions must be mappings")
        kind = exclusion.get("kind")
        reason = exclusion.get("reason")
        if kind == "keys":
            if set(exclusion) != {"kind", "keys", "reason"}:
                raise ValueError("exact rebuild exclusion fields are invalid")
            raw_keys = exclusion["keys"]
            expected_keys: tuple[str, ...] | None = None
        elif kind == "disposition":
            if set(exclusion) != {"disposition", "kind", "keys", "reason"}:
                raise ValueError("category rebuild exclusion fields are invalid")
            raw_keys = exclusion["keys"]
            try:
                disposition = MigrationDisposition(exclusion["disposition"])
            except (TypeError, ValueError) as exc:
                raise ValueError("rebuild exclusion disposition is invalid") from exc
            expected_keys = tuple(
                sorted(
                    entry.entry.key
                    for entry in entries
                    if entry.disposition is disposition
                )
            )
        else:
            raise ValueError("rebuild exclusion kind is invalid")
        if not isinstance(raw_keys, (list, tuple)) or not raw_keys:
            raise ValueError("rebuild exclusion requires a non-empty exact key set")
        keys = tuple(sorted(_compatibility_text(key, "excluded key") for key in raw_keys))
        if len(keys) != len(set(keys)):
            raise ValueError("rebuild exclusion keys must be unique")
        if set(keys) - set(inventory):
            raise ValueError("rebuild exclusion names a key outside the inventory")
        if expected_keys is not None and keys != expected_keys:
            raise ValueError("rebuild exclusion category does not match the inventory")
        if selected_keys.intersection(keys):
            raise ValueError("rebuild exclusions cannot overlap")
        selected_keys.update(keys)
        record: dict[str, object] = {
            "kind": kind,
            "keys": keys,
            "reason": _compatibility_text(reason, "exclusion reason"),
        }
        if kind == "disposition":
            record["disposition"] = exclusion["disposition"]
        normalized.append(MappingProxyType(record))
    return tuple(normalized)


def _rebuild_plan_id(
    *,
    source_identity: AuthorityIdentitySnapshot,
    destination_identity: AuthorityIdentitySnapshot,
    source_revision: int,
    entries: tuple[MigrationEntryAssessment, ...],
    exclusions: tuple[Mapping[str, object], ...],
) -> str:
    """Derive a non-reusable public plan identifier from exact rebuild scope."""
    record = {
        "destination_identity": _identity_record(destination_identity),
        "entries": [
            {
                "generation": entry.entry.generation,
                "key": entry.entry.key,
                "manifest_digest": hashlib.sha256(entry.entry.manifest).hexdigest(),
            }
            for entry in sorted(entries, key=lambda item: item.entry.key)
        ],
        "exclusions": [_thaw_json(item) for item in exclusions],
        "source_identity": _identity_record(source_identity),
        "source_revision": source_revision,
    }
    digest = hashlib.sha256(_canonical_plan_bytes(record)).hexdigest()
    return f"rebuild-{digest[:24]}"


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
    included_totals: MigrationTotals | None = None
    excluded_totals: MigrationTotals | None = None
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
        if self.plan_kind is MigrationPlanKind.REBUILD:
            normalized_exclusions = _normalize_rebuild_exclusions(self.exclusions, self.entries)
        elif self.exclusions:
            raise ValueError("only rebuild plans may exclude inventory entries")
        else:
            normalized_exclusions = ()
        object.__setattr__(self, "exclusions", normalized_exclusions)
        excluded_keys = {
            key for exclusion in normalized_exclusions for key in exclusion["keys"]
        }
        included_entries = tuple(
            entry for entry in self.entries if entry.entry.key not in excluded_keys
        )
        excluded_entries = tuple(
            entry for entry in self.entries if entry.entry.key in excluded_keys
        )
        if self.included_totals is None:
            object.__setattr__(
                self, "included_totals", MigrationTotals.from_entries(included_entries)
            )
        if self.excluded_totals is None:
            object.__setattr__(
                self, "excluded_totals", MigrationTotals.from_entries(excluded_entries)
            )
        if not isinstance(self.included_totals, MigrationTotals) or not isinstance(
            self.excluded_totals, MigrationTotals
        ):
            raise TypeError("rebuild plan selection totals must be MigrationTotals")
        if self.included_totals != MigrationTotals.from_entries(included_entries) or self.excluded_totals != MigrationTotals.from_entries(excluded_entries):
            raise ValueError("rebuild plan selection totals disagree with exclusions")
        if self.plan_kind is MigrationPlanKind.MIGRATION and self.state is not MigrationPlanState.PLANNED:
            raise ValueError("a migration plan must be planned before any mutation")
        if self.plan_kind is MigrationPlanKind.REBUILD and self.state not in {
            MigrationPlanState.PLANNED,
            MigrationPlanState.REFUSED,
        }:
            raise ValueError("a rebuild plan has an invalid state")
        if self.plan_kind is MigrationPlanKind.REFUSED and self.state is not MigrationPlanState.REFUSED:
            raise ValueError("a refused plan must remain non-mutating")
        required_actions = {
            MigrationPlanKind.MIGRATION: ("stage", "verify", "activate"),
            MigrationPlanKind.REFUSED: (),
        }
        if self.plan_kind is MigrationPlanKind.REBUILD:
            required_actions[MigrationPlanKind.REBUILD] = (
                ("stage_rebuild", "verify_rebuild", "accept_rebuild")
                if self.state is MigrationPlanState.PLANNED
                else ("rebuild",)
            )
        if self.intended_actions != required_actions[self.plan_kind]:
            raise ValueError("plan kind has an invalid intended action sequence")
        expected_plan_id = (
            _rebuild_plan_id(
                source_identity=self.source_identity,
                destination_identity=self.destination_identity,
                source_revision=self.source_revision,
                entries=self.entries,
                exclusions=normalized_exclusions,
            )
            if self.plan_kind is MigrationPlanKind.REBUILD
            else self.run_id
        )
        if not self.plan_id:
            object.__setattr__(self, "plan_id", expected_plan_id)
        if self.plan_id != expected_plan_id:
            raise ValueError("plan_id does not bind the exact maintenance scope")
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
        return bool(
            set(self.intended_actions)
            & {
                "stage",
                "verify",
                "activate",
                "rebuild",
                "stage_rebuild",
                "verify_rebuild",
                "accept_rebuild",
            }
        )

    @property
    def excluded_entries(self) -> tuple[MigrationEntryAssessment, ...]:
        """Return only exact inventory members covered by explicit exclusions."""
        excluded_keys = {
            key for exclusion in self.exclusions for key in exclusion["keys"]
        }
        return tuple(entry for entry in self.entries if entry.entry.key in excluded_keys)

    @property
    def included_entries(self) -> tuple[MigrationEntryAssessment, ...]:
        """Return the complete inventory less only exact confirmed exclusions."""
        excluded_keys = {
            key for exclusion in self.exclusions for key in exclusion["keys"]
        }
        return tuple(entry for entry in self.entries if entry.entry.key not in excluded_keys)

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

    @classmethod
    def create_rebuild(
        cls,
        *,
        run_id: str,
        source_identity: AuthorityIdentitySnapshot,
        destination_identity: AuthorityIdentitySnapshot,
        entries: tuple[MigrationEntryAssessment, ...],
        exclusions: tuple[Mapping[str, object], ...],
    ) -> "MigrationPlan":
        """Build a separately actionable rebuild plan from an exact inventory."""
        return cls(
            run_id=run_id,
            source_identity=source_identity,
            destination_identity=destination_identity,
            entries=entries,
            plan_kind=MigrationPlanKind.REBUILD,
            release_window=ReleaseWindow.first_release_baseline(),
            compatibility=_all_supported_compatibility(),
            intended_actions=("stage_rebuild", "verify_rebuild", "accept_rebuild"),
            exclusions=exclusions,
            state=MigrationPlanState.PLANNED,
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
            "excluded_totals": _totals_record(self.excluded_totals),
            "exclusions": [_thaw_json(item) for item in self.exclusions],
            "included_totals": _totals_record(self.included_totals),
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
            "excluded_totals",
            "exclusions",
            "included_totals",
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
                included_totals=_totals_from_record(record["included_totals"]),
                excluded_totals=_totals_from_record(record["excluded_totals"]),
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


def _totals_record(totals: MigrationTotals | None) -> dict[str, object]:
    """Render one complete exact selection aggregate for canonical plans."""
    if not isinstance(totals, MigrationTotals):
        raise TypeError("migration totals are unavailable")
    return {
        "bytes_by_disposition": {
            item.value: totals.bytes_by_disposition[item] for item in MigrationDisposition
        },
        "counts": {item.value: totals.counts[item] for item in MigrationDisposition},
        "total_bytes": totals.total_bytes,
        "total_entries": totals.total_entries,
    }


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
class MigrationRunLimits:
    """Explicit bounded scale policy for one independently recoverable run."""

    max_entries_per_run: int = _DEFAULT_MAX_ENTRIES_PER_RUN
    max_bytes_per_run: int = _DEFAULT_MAX_BYTES_PER_RUN
    max_evidence_bytes_per_run: int = _DEFAULT_MAX_EVIDENCE_BYTES_PER_RUN

    def __post_init__(self) -> None:
        for field_name in (
            "max_entries_per_run",
            "max_bytes_per_run",
            "max_evidence_bytes_per_run",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True)
class MigrationRunPartition:
    """One deterministic, independently actionable bounded portion of an inventory."""

    ordinal: int
    keys: tuple[str, ...]
    entry_count: int
    byte_count: int
    evidence_byte_count: int

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("partition ordinal must be a non-negative integer")
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("partition keys must be a unique non-empty tuple")
        if self.entry_count != len(self.keys):
            raise ValueError("partition entry count disagrees with keys")
        for field_name in ("byte_count", "evidence_byte_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"partition {field_name} must be non-negative")


@dataclass(frozen=True)
class MigrationSplitRequired:
    """Stable non-mutating result for a catalog that exceeds one run's caps."""

    reason: str
    limits: MigrationRunLimits
    partitions: tuple[MigrationRunPartition, ...]

    def __post_init__(self) -> None:
        if self.reason != "split_required":
            raise ValueError("split result reason must be split_required")
        if not isinstance(self.limits, MigrationRunLimits):
            raise TypeError("split result requires MigrationRunLimits")
        if not self.partitions or not all(
            isinstance(partition, MigrationRunPartition) for partition in self.partitions
        ):
            raise ValueError("split result requires bounded partitions")
        if tuple(partition.ordinal for partition in self.partitions) != tuple(
            range(len(self.partitions))
        ):
            raise ValueError("split partitions must have contiguous ordinals")
        if len({key for partition in self.partitions for key in partition.keys}) != sum(
            partition.entry_count for partition in self.partitions
        ):
            raise ValueError("split partitions must not overlap")


@dataclass(frozen=True)
class MigrationStepResult:
    """Minimal result for one completed explicit maintenance action."""

    state: MaintenanceEvidenceState
    completed: bool
    evidence_path: Path


@dataclass(frozen=True)
class AbortReceipt:
    """Result of an explicit unactivated-candidate retirement attempt."""

    run_id: str
    deleted_entries: int
    state: MaintenanceEvidenceState = MaintenanceEvidenceState.ABORTED

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("abort receipt requires a non-empty run_id")
        if type(self.deleted_entries) is not int or self.deleted_entries < 0:
            raise ValueError("abort receipt deleted_entries must be non-negative")
        if self.state not in {
            MaintenanceEvidenceState.STAGING,
            MaintenanceEvidenceState.ABORTED,
        }:
            raise ValueError("abort receipt must record staging or aborted state")


@dataclass(frozen=True)
class PurgeReceipt:
    """Typed post-finalization cleanup outcome that cannot rewrite activation."""

    run_id: str
    purged_entries: int
    pending_entries: int
    cleanup_debt: tuple[str, ...]
    completed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("purge receipt requires a non-empty run_id")
        for field_name in ("purged_entries", "pending_entries"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"purge receipt {field_name} must be non-negative")
        if not isinstance(self.cleanup_debt, tuple) or any(
            not isinstance(item, str) or len(item) != 64 for item in self.cleanup_debt
        ):
            raise ValueError("purge receipt cleanup_debt must contain SHA-256 values")
        if not isinstance(self.completed, bool) or self.completed != (self.pending_entries == 0):
            raise ValueError("purge receipt completion must match pending cleanup work")


class OfflineMigrationService:
    """Coordinate an operator's explicit inspect-to-activate maintenance run.

    A supplied work directory holds authenticated operator evidence only. It
    never becomes a candidate locator, lifecycle state source, or visibility
    authority; the destination ``MigrationAuthority`` owns activation. The
    service supports only declared migration edges and separately confirmed
    handler-backed rebuilds; it does not add an implicit ordinary-open path.
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
        run_limits: MigrationRunLimits | None = None,
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
        self.run_limits = MigrationRunLimits() if run_limits is None else run_limits
        if not isinstance(self.run_limits, MigrationRunLimits):
            raise TypeError("run_limits must be a MigrationRunLimits value")
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
        candidate_batch_references: tuple[str, ...] = (),
        candidate_checkpoint_revision: int | None = None,
        candidate_entry_count: int = 0,
        candidate_byte_count: int = 0,
        activation_receipt: ActivationReceipt | None = None,
        authority_receipts: tuple[str, ...] = (),
        cleanup_debt: tuple[str, ...] = (),
        rebuild_receipt_batches: tuple[RebuildReceiptBatch, ...] = (),
        retired_rebuild_operation_ids: tuple[str, ...] = (),
        completed_output_digests: Mapping[str, str] | None = None,
    ) -> MaintenanceRunEvidence:
        """Build a fully bound evidence record without serializing signing bytes."""
        if candidate_receipt is not None and candidate_entry_count == 0:
            candidate_entry_count = candidate_receipt.entry_count
            candidate_byte_count = candidate_receipt.byte_count
        if candidate_receipt is not None and candidate_checkpoint_revision is None:
            candidate_checkpoint_revision = candidate_receipt.destination_revision
        return MaintenanceRunEvidence(
            evidence_version=2,
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
            candidate_batch_references=candidate_batch_references,
            candidate_checkpoint_revision=candidate_checkpoint_revision,
            candidate_entry_count=candidate_entry_count,
            candidate_byte_count=candidate_byte_count,
            activation_receipt=activation_receipt,
            authority_receipts=authority_receipts,
            cleanup_debt=cleanup_debt,
            rebuild_receipt_batches=rebuild_receipt_batches,
            retired_rebuild_operation_ids=retired_rebuild_operation_ids,
            completed_output_digests=completed_output_digests or {},
        )

    @staticmethod
    def _candidate_progress(evidence: MaintenanceRunEvidence) -> dict[str, object]:
        """Carry signed candidate-progress references across workflow transitions."""
        return {
            "candidate_batch_references": evidence.candidate_batch_references,
            "candidate_checkpoint_revision": evidence.candidate_checkpoint_revision,
            "candidate_entry_count": evidence.candidate_entry_count,
            "candidate_byte_count": evidence.candidate_byte_count,
        }

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

    def _configured_destination_contract(
        self, manifest: BlobManifest
    ) -> tuple[object, StoreVersionDimensions]:
        """Return the destination handler contract without inspecting payload bytes."""
        handler = self.destination.handlers.get_handler_by_type(manifest.handler_type)
        payload_version = getattr(handler, "payload_format_version", None)
        if type(payload_version) is not int or payload_version < 1:
            raise ValueError("destination handler declares an invalid payload contract")
        return handler, StoreVersionDimensions(payload_format_version=payload_version)

    def _matching_compatibility_edges(
        self,
        manifest: BlobManifest,
        destination_versions: StoreVersionDimensions,
    ) -> tuple[MigrationCompatibilityEdge, ...]:
        """Return only edges whose source and configured destination both match."""
        return tuple(
            edge
            for edge in self.compatibility_edges
            if edge.supports(manifest.versions, destination_versions)
        )

    def _candidate_blob_id(self, manifest: BlobManifest) -> str:
        """Mint an opaque attempt-local identifier for one immutable candidate.

        A payload written before the authority acknowledges its descriptor has
        no recovery identity.  Retrying therefore uses a fresh locator rather
        than deriving and reusing the earlier payload's name.
        """
        material = "\x00".join(
            (self.run_id, manifest.key, manifest.generation, manifest.digest, uuid4().hex)
        ).encode("utf-8")
        return f"{hashlib.sha256(material).hexdigest()}{''.join(Path(manifest.locator).suffixes)}"

    def _candidate_locator(self, candidate_id: str) -> str:
        """Derive the exact backend locator without treating a candidate path as authority."""
        backend = self.destination.payload_backend
        remote_locator = getattr(backend, "migration_candidate_locator", None)
        if callable(remote_locator):
            return remote_locator(run_id=self.run_id, candidate_id=candidate_id)
        base_dir = getattr(backend, "base_dir", None)
        shard_chars = getattr(backend, "shard_chars", None)
        if isinstance(base_dir, Path) and type(shard_chars) is int:
            shard = candidate_id[:shard_chars] if shard_chars else ""
            return str(base_dir / shard / candidate_id) if shard else str(base_dir / candidate_id)
        return candidate_id

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
                source_matches = tuple(
                    edge
                    for edge in self.compatibility_edges
                    if edge.source == manifest.versions
                )
                try:
                    destination_handler, destination_versions = (
                        self._configured_destination_contract(manifest)
                    )
                    matching_edges = self._matching_compatibility_edges(
                        manifest, destination_versions
                    )
                    self.source.handlers.resolve_payload_contract(
                        manifest.handler_type,
                        manifest.payload_format,
                        manifest.payload_format_version,
                    )
                except (ValueError, CacheManifestUnsupportedVersionError):
                    matching_edges = ()
                    destination_handler = None
                if len(matching_edges) != 1:
                    disposition = MigrationDisposition.REBUILDABLE
                    reason = (
                        MigrationReason.DESTINATION_MISMATCH
                        if source_matches
                        else MigrationReason.SOURCE_MISMATCH
                    )
                elif (
                    manifest.payload_format == destination_handler.payload_format
                    and manifest.payload_format_version
                    == destination_handler.payload_format_version
                ):
                    disposition = MigrationDisposition.MIGRATABLE
                    reason = MigrationReason.COMPATIBLE_EDGE
                else:
                    try:
                        self.source.handlers.resolve_payload_transformation(
                            manifest.handler_type,
                            manifest.payload_format,
                            manifest.payload_format_version,
                            destination_handler.payload_format,
                            destination_handler.payload_format_version,
                        )
                    except (ValueError, CacheManifestUnsupportedVersionError):
                        disposition = MigrationDisposition.REBUILDABLE
                        reason = MigrationReason.NON_EXECUTABLE_TRANSFORMATION
                    else:
                        disposition = MigrationDisposition.MIGRATABLE
                        reason = MigrationReason.DIRECTED_EDGE
                assessments.append(
                    MigrationEntryAssessment(
                        entry=entry,
                        disposition=disposition,
                        reason=reason,
                        catalog_values=dict(manifest.catalog_values),
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

    @staticmethod
    def _candidate_evidence_size(assessment: MigrationEntryAssessment) -> int:
        """Bound an authority-held target descriptor without rendering its manifest."""
        entry = assessment.entry
        return (
            len(entry.manifest)
            + len(entry.key.encode("utf-8"))
            + len(entry.generation.encode("utf-8"))
            + len(entry.locator.encode("utf-8"))
            + 256
        )

    def _split_required(
        self, assessments: tuple[MigrationEntryAssessment, ...]
    ) -> MigrationSplitRequired | None:
        """Partition an oversized catalog before any external candidate write."""
        total_bytes = sum(assessment.entry.byte_size for assessment in assessments)
        total_evidence_bytes = sum(
            self._candidate_evidence_size(assessment) for assessment in assessments
        )
        limits = self.run_limits
        if (
            len(assessments) <= limits.max_entries_per_run
            and total_bytes <= limits.max_bytes_per_run
            and total_evidence_bytes <= limits.max_evidence_bytes_per_run
        ):
            return None

        partitions: list[MigrationRunPartition] = []
        current: list[MigrationEntryAssessment] = []
        current_bytes = 0
        current_evidence_bytes = 0
        for assessment in sorted(assessments, key=lambda item: item.entry.key):
            entry_bytes = assessment.entry.byte_size
            evidence_bytes = self._candidate_evidence_size(assessment)
            if (
                entry_bytes > limits.max_bytes_per_run
                or evidence_bytes > limits.max_evidence_bytes_per_run
            ):
                raise ValueError("one migration entry exceeds every independently recoverable run")
            exceeds = (
                len(current) + 1 > limits.max_entries_per_run
                or current_bytes + entry_bytes > limits.max_bytes_per_run
                or current_evidence_bytes + evidence_bytes
                > limits.max_evidence_bytes_per_run
            )
            if current and exceeds:
                partitions.append(
                    MigrationRunPartition(
                        ordinal=len(partitions),
                        keys=tuple(item.entry.key for item in current),
                        entry_count=len(current),
                        byte_count=current_bytes,
                        evidence_byte_count=current_evidence_bytes,
                    )
                )
                current = []
                current_bytes = 0
                current_evidence_bytes = 0
            current.append(assessment)
            current_bytes += entry_bytes
            current_evidence_bytes += evidence_bytes
        if current:
            partitions.append(
                MigrationRunPartition(
                    ordinal=len(partitions),
                    keys=tuple(item.entry.key for item in current),
                    entry_count=len(current),
                    byte_count=current_bytes,
                    evidence_byte_count=current_evidence_bytes,
                )
            )
        return MigrationSplitRequired(
            reason="split_required", limits=limits, partitions=tuple(partitions)
        )

    def plan(self, inspection: MigrationInspection) -> MigrationPlan | MigrationSplitRequired:
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
        split = self._split_required(inspection.assessments)
        if split is not None:
            return split
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

    def create_rebuild_plan(
        self,
        inspection: MigrationInspection,
        *,
        exclusions: tuple[RebuildExclusion, ...] = (),
    ) -> MigrationPlan | MigrationSplitRequired:
        """Create an explicit include-all rebuild plan from one fixed inspection.

        Unlike :meth:`plan`, this method never makes a same-backend physical
        migration eligible.  It records a complete inventory and resolves any
        category selectors into exact keys before an operator can confirm it.
        """
        if not isinstance(inspection, MigrationInspection):
            raise TypeError("inspection must be a MigrationInspection")
        if not isinstance(exclusions, tuple) or not all(
            isinstance(exclusion, RebuildExclusion) for exclusion in exclusions
        ):
            raise TypeError("rebuild exclusions must contain RebuildExclusion values")
        current_source, current_destination = self._revalidate_identities()
        if (
            current_source != inspection.source_identity
            or current_destination != inspection.destination_identity
        ):
            raise CacheBlobMigrationPlanStaleError(
                "inspection is stale; reinspection is required",
                context={"operation": "migration.create_rebuild_plan", "run_id": self.run_id},
            )
        evidence = self._expect_evidence(MaintenanceEvidenceState.INSPECTED, plan_digest="")
        if evidence.source_identity != inspection.source_identity:
            raise CacheBlobMigrationEvidenceMismatchError(
                "inspection evidence does not corroborate the source",
                context={"operation": "migration.create_rebuild_plan", "run_id": self.run_id},
            )
        split = self._split_required(inspection.assessments)
        if split is not None:
            return split
        return MigrationPlan.create_rebuild(
            run_id=self.run_id,
            source_identity=inspection.source_identity,
            destination_identity=inspection.destination_identity,
            entries=inspection.assessments,
            exclusions=tuple(exclusion.resolve(inspection.assessments) for exclusion in exclusions),
        )

    def rebuild_confirmation(self, plan: MigrationPlan) -> str:
        """Return the exact operator token for one immutable rebuild plan."""
        if not isinstance(plan, MigrationPlan) or plan.run_id != self.run_id:
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild plan does not belong to this explicit run",
                context={"operation": "migration.rebuild_confirmation", "run_id": self.run_id},
            )
        if plan.plan_kind is not MigrationPlanKind.REBUILD or plan.state is not MigrationPlanState.PLANNED:
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild confirmation requires an actionable rebuild plan",
                context={"operation": "migration.rebuild_confirmation", "run_id": self.run_id},
            )
        if not hmac.compare_digest(plan.digest, plan._expected_digest()):
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild plan digest is invalid",
                context={"operation": "migration.rebuild_confirmation", "run_id": self.run_id},
            )
        material = "\x00".join(("cacheness-rebuild-confirmation-v1", plan.plan_id, plan.digest))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def confirm_rebuild(self, plan: MigrationPlan, *, confirmation: str) -> MigrationStepResult:
        """Bind an exact operator confirmation into authenticated offline evidence."""
        self._validate_rebuild_plan(plan)
        expected_confirmation = self.rebuild_confirmation(plan)
        if not isinstance(confirmation, str) or not hmac.compare_digest(
            confirmation, expected_confirmation
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild confirmation does not bind this exact plan",
                context={"operation": "migration.confirm_rebuild", "run_id": self.run_id},
            )
        self._expect_evidence(MaintenanceEvidenceState.INSPECTED, plan_digest="")
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.PLANNED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=("inspect", "confirm_rebuild"),
                completed_output_digests={
                    "confirm_rebuild": expected_confirmation,
                },
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.PLANNED, True, self.evidence_path)

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

    def _validate_rebuild_plan(
        self, plan: MigrationPlan, *, require_destination_revision: bool = False
    ) -> None:
        """Reject physical-migration plans before a rebuild action can begin."""
        if not isinstance(plan, MigrationPlan) or plan.run_id != self.run_id:
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild plan does not belong to this explicit run",
                context={"operation": "migration.rebuild", "run_id": self.run_id},
            )
        if not hmac.compare_digest(plan.digest, plan._expected_digest()):
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild plan digest is invalid",
                context={"operation": "migration.rebuild", "run_id": self.run_id},
            )
        if plan.plan_kind is not MigrationPlanKind.REBUILD or plan.state is not MigrationPlanState.PLANNED:
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild action requires an actionable rebuild plan",
                context={"operation": "migration.rebuild", "run_id": self.run_id},
            )
        source_identity = self._source_authority.identity_snapshot()
        destination_identity = self._destination_authority.identity_snapshot()
        if source_identity != plan.source_identity or (
            destination_identity.store_id != plan.destination_identity.store_id
            or destination_identity.authority_kind != plan.destination_identity.authority_kind
            or destination_identity.capability != plan.destination_identity.capability
            or destination_identity.schema_version != plan.destination_identity.schema_version
            or (
                require_destination_revision
                and destination_identity.revision != plan.destination_identity.revision
            )
        ):
            raise CacheBlobMigrationPlanStaleError(
                "rebuild plan is stale; source or destination identity changed",
                context={"operation": "migration.rebuild", "run_id": self.run_id},
            )

    def _expect_rebuild_evidence(
        self, state: MaintenanceEvidenceState, *, plan: MigrationPlan
    ) -> MaintenanceRunEvidence:
        """Revalidate source truth while allowing only this run's destination writes."""
        evidence = self.read_evidence()
        if evidence.state is not state or evidence.plan_digest != plan.digest:
            raise CacheBlobMigrationEvidenceMismatchError(
                f"maintenance evidence is not ready for rebuild {state.value}",
                context={"operation": "migration.rebuild_evidence", "run_id": self.run_id},
            )
        self._validate_rebuild_plan(plan)
        if (
            evidence.source_identity != plan.source_identity
            or evidence.source_revision != plan.source_identity.revision
            or evidence.destination_identity != plan.destination_identity
            or evidence.destination_revision != plan.destination_identity.revision
            or evidence.acknowledgement
            != self._acknowledgement(plan.digest, plan.source_identity)
        ):
            raise CacheBlobMigrationPlanStaleError(
                "rebuild evidence source or destination is stale; reinspection is required",
                context={"operation": "migration.rebuild_evidence", "run_id": self.run_id},
            )
        return evidence

    @staticmethod
    def _rebuild_output_digest(receipts: tuple[BlobReceipt, ...]) -> str:
        """Bind only exact, projection-free destination ownership to a digest."""
        record = [
            {
                "catalog_revision": receipt.catalog_revision,
                "expectation": {
                    "generation": receipt.expectation.generation,
                    "lineage": receipt.expectation.lineage,
                    "manifest_digest": receipt.expectation.manifest_digest,
                    "revision": receipt.expectation.revision,
                },
                "generation": receipt.generation,
                "key": receipt.key,
                "locator": receipt.locator,
                "operation_id": receipt.operation_id,
            }
            for receipt in receipts
        ]
        return hashlib.sha256(_canonical_plan_bytes({"receipts": record})).hexdigest()

    @staticmethod
    def _flatten_rebuild_receipts(
        evidence: MaintenanceRunEvidence,
    ) -> tuple[BlobReceipt, ...]:
        """Return the authenticated ordered receipt prefix, not discovered state."""
        return tuple(
            receipt
            for batch in evidence.rebuild_receipt_batches
            for receipt in batch.receipts
        )

    @staticmethod
    def _rebuild_progress(
        evidence: MaintenanceRunEvidence,
        *,
        retired_rebuild_operation_ids: tuple[str, ...] | None = None,
    ) -> dict[str, object]:
        """Carry only the bounded authenticated rebuild progress fields forward."""
        return {
            "rebuild_receipt_batches": evidence.rebuild_receipt_batches,
            "retired_rebuild_operation_ids": (
                evidence.retired_rebuild_operation_ids
                if retired_rebuild_operation_ids is None
                else retired_rebuild_operation_ids
            ),
        }

    @staticmethod
    def _rebuild_cleanup_debt(receipt: BlobReceipt) -> str:
        """Encode one exact external cleanup obligation for the evidence ledger."""
        return (
            f"rebuild:{receipt.operation_id}:{receipt.key}:"
            f"{receipt.generation}:{receipt.locator}"
        )

    def _rebuild_operation_id(
        self,
        plan: MigrationPlan,
        assessment: MigrationEntryAssessment,
        entry_ordinal: int,
    ) -> str:
        """Derive the one non-reusable authority key for a planned rebuild entry."""
        target_contract_digest = hashlib.sha256(
            _canonical_plan_bytes(
                {
                    "catalog_values": _thaw_json(assessment.catalog_values),
                    "destination_authority": plan.destination_identity.authority_kind,
                    "destination_capability": plan.destination_identity.capability,
                    "destination_schema_version": plan.destination_identity.schema_version,
                    "destination_store_id": plan.destination_identity.store_id,
                }
            )
        ).hexdigest()
        return hashlib.sha256(
            _canonical_plan_bytes(
                {
                    "destination_revision": plan.destination_identity.revision,
                    "destination_store_id": plan.destination_identity.store_id,
                    "entry_ordinal": entry_ordinal,
                    "key": assessment.entry.key,
                    "plan_digest": plan.digest,
                    "run_id": plan.run_id,
                    "source_manifest_digest": hashlib.sha256(
                        assessment.entry.manifest
                    ).hexdigest(),
                    "source_payload_digest": assessment.entry.payload_digest,
                    "target_contract_digest": target_contract_digest,
                    "version": "cacheness-rebuild-operation-v1",
                }
            )
        ).hexdigest()

    def _validate_rebuild_receipts(
        self,
        plan: MigrationPlan,
        evidence: MaintenanceRunEvidence,
        *,
        require_complete: bool,
    ) -> tuple[BlobReceipt, ...]:
        """Authenticate the recorded prefix against exact authority results only."""
        receipts = self._flatten_rebuild_receipts(evidence)
        if len(receipts) > len(plan.included_entries) or (
            require_complete and len(receipts) != len(plan.included_entries)
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild receipt count does not match the bounded plan",
                context={"operation": "migration.rebuild_receipts", "run_id": self.run_id},
            )
        for entry_ordinal, receipt in enumerate(receipts):
            assessment = plan.included_entries[entry_ordinal]
            operation_id = self._rebuild_operation_id(plan, assessment, entry_ordinal)
            replay = self._destination_authority.read_mutation(operation_id)
            if (
                receipt.operation_id != operation_id
                or receipt.key != assessment.entry.key
                or dict(receipt.projections)
                or replay is None
                or replay.state != "promoted"
                or replay.promotion is None
            ):
                raise CacheBlobMigrationEvidenceMismatchError(
                    "rebuild receipt is not an exact promoted authority result",
                    context={"operation": "migration.rebuild_receipts", "run_id": self.run_id},
                )
            promoted = replay.promotion.entry
            if (
                receipt.generation != promoted.generation
                or receipt.locator != promoted.locator
                or receipt.expectation != promoted.expectation
                or receipt.catalog_revision != (promoted.expectation.revision or 0)
            ):
                raise CacheBlobMigrationEvidenceMismatchError(
                    "rebuild receipt does not corroborate authority ownership",
                    context={"operation": "migration.rebuild_receipts", "run_id": self.run_id},
                )
            current = self.destination.get_entry_info(receipt.key)
            if (
                current is None
                or current.generation != receipt.generation
                or current.locator != receipt.locator
                or current.expectation != receipt.expectation
            ):
                raise CacheBlobMigrationEvidenceMismatchError(
                    "rebuild receipt no longer matches the exact destination entry",
                    context={"operation": "migration.rebuild_receipts", "run_id": self.run_id},
                )
        return receipts

    def _reach_rebuild_fault(self, boundary: str) -> None:
        """Provide a process-loss test boundary outside lifecycle ownership."""
        hook = getattr(self, "_rebuild_fault_hook", None)
        if callable(hook):
            hook(boundary)

    def _stage_rebuild_entry(
        self,
        plan: MigrationPlan,
        assessment: MigrationEntryAssessment,
        entry_ordinal: int,
    ) -> BlobReceipt:
        """Read one authenticated source entry and replay one authority operation."""
        operation_id = self._rebuild_operation_id(plan, assessment, entry_ordinal)
        replay = self._destination_authority.read_mutation(operation_id)
        if replay is None and self.destination.get_entry_info(assessment.entry.key) is not None:
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild destination key exists without this operation record",
                context={"operation": "migration.stage_rebuild", "run_id": self.run_id},
            )
        with self.source.open_entry(assessment.entry.key) as source_entry:
            if (
                source_entry is None
                or source_entry.generation != assessment.entry.generation
                or source_entry.locator != assessment.entry.locator
            ):
                raise CacheBlobMigrationPlanStaleError(
                    "rebuild source entry changed; reinspection is required",
                    context={"operation": "migration.stage_rebuild", "run_id": self.run_id},
                )
            catalog = source_entry.metadata.get("catalog")
            if not isinstance(catalog, Mapping) or not isinstance(catalog.get("values"), Mapping):
                raise CacheBlobMigrationEvidenceMismatchError(
                    "authenticated source entry has invalid catalog values",
                    context={"operation": "migration.stage_rebuild", "run_id": self.run_id},
                )
            catalog_values = dict(catalog["values"])
            if catalog_values != dict(assessment.catalog_values):
                raise CacheBlobMigrationPlanStaleError(
                    "rebuild source catalog changed; reinspection is required",
                    context={"operation": "migration.stage_rebuild", "run_id": self.run_id},
                )
            value = source_entry.read()
        self._validate_rebuild_plan(plan)
        try:
            receipt = self.destination._put_entry_canonical_for_maintenance(
                value,
                key=assessment.entry.key,
                catalog_values=catalog_values,
                operation_id=operation_id,
            )
        except Exception as original_error:
            # A catchable lost response can be retried only when the existing
            # lifecycle authority already owns this exact operation identity.
            if self._destination_authority.read_mutation(operation_id) is None:
                raise
            try:
                receipt = self.destination._put_entry_canonical_for_maintenance(
                    value,
                    key=assessment.entry.key,
                    catalog_values=catalog_values,
                    operation_id=operation_id,
                )
            except Exception as retry_error:
                raise retry_error from original_error
        if (
            receipt.operation_id != operation_id
            or receipt.key != assessment.entry.key
            or dict(receipt.projections)
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "canonical rebuild result is not the requested projection-free receipt",
                context={"operation": "migration.stage_rebuild", "run_id": self.run_id},
            )
        return receipt

    def _abort_rebuild_after_failure(
        self,
        plan: MigrationPlan,
        evidence: MaintenanceRunEvidence,
    ) -> None:
        """Retire only receipt-owned outputs, preserving conflicts as exact debt."""
        receipts = self._flatten_rebuild_receipts(evidence)
        retired = list(evidence.retired_rebuild_operation_ids)
        cleanup_debt = list(evidence.cleanup_debt)
        for receipt in reversed(receipts):
            if receipt.operation_id in retired:
                continue
            try:
                current = self.destination.get_entry_info(receipt.key)
                if current is None:
                    retired.append(receipt.operation_id)
                elif (
                    current.generation != receipt.generation
                    or current.locator != receipt.locator
                    or current.expectation != receipt.expectation
                ):
                    cleanup_debt.append(self._rebuild_cleanup_debt(receipt))
                else:
                    self.destination.delete(receipt.key, expected=receipt.expectation)
                    retired.append(receipt.operation_id)
            except Exception:
                cleanup_debt.append(self._rebuild_cleanup_debt(receipt))
            cleanup_debt = list(dict.fromkeys(cleanup_debt))
            retired = list(dict.fromkeys(retired))
            evidence = self._write_evidence(
                self._new_evidence(
                    state=evidence.state,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=evidence.completed_steps,
                    authority_receipts=evidence.authority_receipts,
                    cleanup_debt=tuple(cleanup_debt),
                    **self._rebuild_progress(
                        evidence,
                        retired_rebuild_operation_ids=tuple(retired),
                    ),
                    completed_output_digests=evidence.completed_output_digests,
                )
            )
        covered = set(retired)
        covered.update(
            debt.split(":", 2)[1]
            for debt in cleanup_debt
            if debt.startswith("rebuild:")
        )
        if any(receipt.operation_id not in covered for receipt in receipts):
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild cleanup lacks an attributed terminal outcome",
                context={"operation": "migration.abort_rebuild", "run_id": self.run_id},
            )
        output_digest = self._rebuild_output_digest(receipts)
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.ABORTED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=(*evidence.completed_steps, "abort_rebuild"),
                authority_receipts=(*evidence.authority_receipts, output_digest),
                cleanup_debt=tuple(cleanup_debt),
                **self._rebuild_progress(
                    evidence,
                    retired_rebuild_operation_ids=tuple(retired),
                ),
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "abort_rebuild": output_digest,
                },
            )
        )

    def stage_rebuild(self, plan: MigrationPlan) -> MigrationStepResult:
        """Rebuild every included value through verified source and destination APIs.

        ``BlobStore.open_entry`` authenticates the source manifest, verifies a
        private snapshot's digest and size, and only then permits handler
        deserialization.  ``BlobStore.put_entry`` remains the sole destination
        lifecycle publisher; this coordinator never writes native payload
        bytes or authority rows directly.
        """
        current = self.read_evidence()
        self._validate_rebuild_plan(
            plan, require_destination_revision=current.state is MaintenanceEvidenceState.PLANNED
        )
        if self._split_required(plan.included_entries) is not None:
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild plan exceeds one bounded run and must be split before staging",
                context={"operation": "migration.stage_rebuild", "run_id": self.run_id},
            )
        if current.state is MaintenanceEvidenceState.PLANNED:
            evidence = self._expect_rebuild_evidence(
                MaintenanceEvidenceState.PLANNED, plan=plan
            )
            evidence = self._write_evidence(
                self._new_evidence(
                    state=MaintenanceEvidenceState.REBUILDING,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=("inspect", "confirm_rebuild"),
                    completed_output_digests=dict(evidence.completed_output_digests),
                )
            )
        elif current.state is MaintenanceEvidenceState.REBUILDING:
            evidence = self._expect_rebuild_evidence(
                MaintenanceEvidenceState.REBUILDING, plan=plan
            )
        else:
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence is not ready to stage this rebuild",
                context={"operation": "migration.stage_rebuild", "run_id": self.run_id},
            )

        receipts = self._validate_rebuild_receipts(
            plan, evidence, require_complete=False
        )
        try:
            for entry_ordinal, assessment in enumerate(plan.included_entries):
                if entry_ordinal < len(receipts):
                    continue
                receipt = self._stage_rebuild_entry(plan, assessment, entry_ordinal)
                # This is a testable crash boundary.  It deliberately sits
                # after the sole lifecycle authority's commit and before the
                # separate maintenance evidence checkpoint.
                self._reach_rebuild_fault(
                    "rebuild.destination_committed_before_receipt_checkpoint"
                )
                batch = RebuildReceiptBatch.create(
                    batch_ordinal=len(evidence.rebuild_receipt_batches),
                    first_entry_ordinal=entry_ordinal,
                    receipts=(receipt,),
                    byte_count=assessment.entry.byte_size,
                )
                evidence = self._write_evidence(
                    self._new_evidence(
                        state=MaintenanceEvidenceState.REBUILDING,
                        plan_digest=plan.digest,
                        source_identity=plan.source_identity,
                        destination_identity=plan.destination_identity,
                        completed_steps=evidence.completed_steps,
                        authority_receipts=evidence.authority_receipts,
                        cleanup_debt=evidence.cleanup_debt,
                        rebuild_receipt_batches=(
                            *evidence.rebuild_receipt_batches,
                            batch,
                        ),
                        retired_rebuild_operation_ids=(
                            evidence.retired_rebuild_operation_ids
                        ),
                        completed_output_digests=evidence.completed_output_digests,
                    )
                )
                receipts = (*receipts, receipt)
        except Exception:
            self._abort_rebuild_after_failure(plan, evidence)
            raise

        output_digest = self._rebuild_output_digest(receipts)
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.REBUILD_STAGED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=("inspect", "confirm_rebuild", "stage_rebuild"),
                authority_receipts=(*evidence.authority_receipts, output_digest),
                cleanup_debt=evidence.cleanup_debt,
                **self._rebuild_progress(evidence),
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "stage_rebuild": output_digest,
                },
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.REBUILD_STAGED, True, self.evidence_path)

    def verify_rebuild(self, plan: MigrationPlan) -> MigrationStepResult:
        """Verify the exact included destination set before explicit acceptance."""
        self._validate_rebuild_plan(plan)
        current = self.read_evidence()
        if current.state is MaintenanceEvidenceState.REBUILD_STAGED:
            evidence = self._expect_rebuild_evidence(
                MaintenanceEvidenceState.REBUILD_STAGED, plan=plan
            )
            evidence = self._write_evidence(
                self._new_evidence(
                    state=MaintenanceEvidenceState.REBUILD_VERIFYING,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=evidence.completed_steps,
                    authority_receipts=evidence.authority_receipts,
                    cleanup_debt=evidence.cleanup_debt,
                    **self._rebuild_progress(evidence),
                    completed_output_digests=evidence.completed_output_digests,
                )
            )
        elif current.state is MaintenanceEvidenceState.REBUILD_VERIFYING:
            evidence = self._expect_rebuild_evidence(
                MaintenanceEvidenceState.REBUILD_VERIFYING, plan=plan
            )
        else:
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence is not ready to verify this rebuild",
                context={"operation": "migration.verify_rebuild", "run_id": self.run_id},
            )

        receipts = self._validate_rebuild_receipts(
            plan, evidence, require_complete=True
        )
        try:
            for assessment, receipt in zip(plan.included_entries, receipts, strict=True):
                with self.destination.open_entry(assessment.entry.key) as destination_entry:
                    if (
                        destination_entry is None
                        or destination_entry.generation != receipt.generation
                        or destination_entry.locator != receipt.locator
                        or destination_entry.expectation != receipt.expectation
                    ):
                        raise CacheBlobMigrationEvidenceMismatchError(
                            "rebuild destination entry is absent",
                            context={
                                "operation": "migration.verify_rebuild",
                                "run_id": self.run_id,
                            },
                        )
                    catalog = destination_entry.metadata.get("catalog")
                    if not isinstance(catalog, Mapping) or dict(catalog.get("values", {})) != dict(
                        assessment.catalog_values
                    ):
                        raise CacheBlobMigrationEvidenceMismatchError(
                            "rebuild destination catalog values do not match the source",
                            context={
                                "operation": "migration.verify_rebuild",
                                "run_id": self.run_id,
                            },
                        )
        except Exception:
            self._abort_rebuild_after_failure(plan, evidence)
            raise
        verify_digest = hashlib.sha256(
            _canonical_plan_bytes(
                {
                    "catalog": [
                        _thaw_json(assessment.catalog_values)
                        for assessment in plan.included_entries
                    ],
                    "receipt_digest": self._rebuild_output_digest(receipts),
                }
            )
        ).hexdigest()
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.REBUILD_VERIFIED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=(*evidence.completed_steps, "verify_rebuild"),
                authority_receipts=evidence.authority_receipts,
                cleanup_debt=evidence.cleanup_debt,
                **self._rebuild_progress(evidence),
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "verify_rebuild": verify_digest,
                },
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.REBUILD_VERIFIED, True, self.evidence_path)

    def accept_rebuild(self, plan: MigrationPlan) -> MigrationStepResult:
        """Record explicit acceptance after complete destination verification.

        BlobStore has already committed each destination entry through its own
        lifecycle authority.  This evidence checkpoint is corroborative
        offline maintenance state, never a second visibility authority.
        """
        self._validate_rebuild_plan(plan)
        evidence = self._expect_rebuild_evidence(
            MaintenanceEvidenceState.REBUILD_VERIFIED, plan=plan
        )
        self._validate_rebuild_receipts(plan, evidence, require_complete=True)
        accepted_digest = hashlib.sha256(
            f"{plan.plan_id}:{plan.digest}:accepted".encode("utf-8")
        ).hexdigest()
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.REBUILD_ACCEPTED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=(*evidence.completed_steps, "accept_rebuild"),
                authority_receipts=(*evidence.authority_receipts, accepted_digest),
                cleanup_debt=evidence.cleanup_debt,
                **self._rebuild_progress(evidence),
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "accept_rebuild": accepted_digest,
                },
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.REBUILD_ACCEPTED, True, self.evidence_path)

    def _authority_candidate_entries(
        self, plan: MigrationPlan
    ) -> tuple[AuthorityInventoryEntry, ...]:
        """Load only descriptors that the destination authority has attributed."""
        loader = getattr(self._destination_authority, "candidate_entries_for_run", None)
        if not callable(loader):
            raise CacheBlobMigrationEvidenceMismatchError(
                "destination authority cannot attest migration candidate progress",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        observed = loader(run_id=self.run_id)
        if not isinstance(observed, tuple) or not all(
            isinstance(entry, AuthorityInventoryEntry) for entry in observed
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "destination authority returned malformed candidate descriptors",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        expected = {assessment.entry.key: assessment.entry for assessment in plan.assessments}
        by_key = {entry.key: entry for entry in observed}
        if len(by_key) != len(observed) or set(by_key) - set(expected):
            raise CacheBlobMigrationEvidenceMismatchError(
                "authority candidate descriptors are not an exact plan subset",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        ordered: list[AuthorityInventoryEntry] = []
        for assessment in plan.assessments:
            candidate = by_key.get(assessment.entry.key)
            if candidate is None:
                continue
            manifest = self._authenticated_manifest(candidate.manifest, role="candidate")
            source = assessment.entry
            if (
                candidate.generation != source.generation
                or candidate.payload_digest != source.payload_digest
                or candidate.byte_size != source.byte_size
                or manifest.key != candidate.key
                or manifest.generation != candidate.generation
                or manifest.locator != candidate.locator
                or manifest.digest != candidate.payload_digest
                or manifest.byte_size != candidate.byte_size
            ):
                raise CacheBlobMigrationEvidenceMismatchError(
                    "authority candidate descriptor does not bind the planned source entry",
                    context={"operation": "migration.resume", "run_id": self.run_id},
                )
            ordered.append(candidate)
        return tuple(ordered)

    def _candidate_receipt(
        self, plan: MigrationPlan, entries: tuple[AuthorityInventoryEntry, ...]
    ) -> VerifiedCandidateReceipt:
        """Bind the authority-attributed candidate subset to this immutable plan."""
        return VerifiedCandidateReceipt(
            run_id=self.run_id,
            plan_digest=plan.digest,
            source_identity=plan.source_identity,
            source_revision=plan.source_identity.revision,
            destination_identity=plan.destination_identity,
            destination_revision=plan.destination_identity.revision,
            candidate_digest=candidate_digest(entries),
            entry_count=len(entries),
            byte_count=sum(entry.byte_size for entry in entries),
        )

    @staticmethod
    def _batch_receipt(
        plan: MigrationPlan,
        entry: AuthorityInventoryEntry,
        entry_ordinal: int,
    ) -> CandidateBatchReceipt:
        """Describe one acknowledged entry without storing payload data in evidence."""
        descriptor = CandidateEntryReceipt(
            entry_ordinal=entry_ordinal,
            key=entry.key,
            generation=entry.generation,
            locator=entry.locator,
            target_manifest=entry.manifest,
            payload_digest=entry.payload_digest,
            byte_size=entry.byte_size,
        )
        return CandidateBatchReceipt(
            run_id=plan.run_id,
            plan_digest=plan.digest,
            batch_ordinal=entry_ordinal,
            first_entry_ordinal=entry_ordinal,
            past_last_entry_ordinal=entry_ordinal + 1,
            destination_identity=plan.destination_identity,
            destination_revision=plan.destination_identity.revision,
            entries=(descriptor,),
            candidate_digest=candidate_digest((entry,)),
            entry_count=1,
            byte_count=entry.byte_size,
        )

    def stage(self, plan: MigrationPlan) -> MigrationStepResult:
        """Copy payloads and checkpoint each only after authority attribution."""
        self._validate_plan(plan)
        if plan.plan_kind is not MigrationPlanKind.MIGRATION:
            raise CacheBlobMigrationEvidenceMismatchError(
                "rebuild plans must use the explicit rebuild workflow",
                context={"operation": "migration.stage", "run_id": self.run_id},
            )
        current = self.read_evidence()
        if current.state is MaintenanceEvidenceState.PLANNED:
            self._expect_evidence(MaintenanceEvidenceState.PLANNED, plan_digest=plan.digest)
            current = self._write_evidence(
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
            current = self._expect_evidence(
                MaintenanceEvidenceState.STAGING, plan_digest=plan.digest
            )
        else:
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence is not ready to stage this plan",
                context={"operation": "migration.stage", "run_id": self.run_id},
            )

        candidates = list(self._authority_candidate_entries(plan))
        batch_references = list(current.candidate_batch_references)
        observed_batches = tuple(
            self._batch_receipt(plan, entry, ordinal)
            for ordinal, entry in enumerate(candidates)
        )
        observed_references = tuple(batch.candidate_digest for batch in observed_batches)
        if (
            len(batch_references) > len(candidates)
            or tuple(batch_references) != observed_references[: len(batch_references)]
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "staging evidence and authority candidate progress disagree",
                context={"operation": "migration.stage", "run_id": self.run_id},
            )
        if len(batch_references) != len(candidates):
            batch_references = list(observed_references)
            self._write_evidence(
                self._new_evidence(
                    state=MaintenanceEvidenceState.STAGING,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=("inspect", "plan"),
                    candidate_batch_references=tuple(batch_references),
                    candidate_checkpoint_revision=plan.destination_identity.revision,
                    candidate_entry_count=len(candidates),
                    candidate_byte_count=sum(entry.byte_size for entry in candidates),
                    cleanup_debt=current.cleanup_debt,
                    completed_output_digests={"plan": plan.digest},
                )
            )
        expected = {assessment.entry.key: assessment for assessment in plan.assessments}
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
                if snapshot.key in {entry.key for entry in candidates}:
                    continue
                manifest = self._authenticated_manifest(snapshot.manifest, role="source")
                source_io = self.source._materialize_authority_store()
                with source_io.open_snapshot(
                    manifest.locator, dict(manifest.handler_metadata)
                ) as source_snapshot:
                    payload = source_snapshot.path.read_bytes()
                if (
                    hashlib.sha256(payload).hexdigest() != manifest.digest
                    or len(payload) != manifest.byte_size
                ):
                    raise ValueError("source payload fails authenticated integrity verification")
                candidate_id = self._candidate_blob_id(manifest)
                candidate_locator = self._candidate_locator(candidate_id)
                self._revalidate_identities(plan)
                destination_io = self.destination._materialize_authority_store()
                remote_candidate_writer = getattr(destination_io, "write_migration_candidate", None)
                if callable(remote_candidate_writer):
                    remote_receipt = remote_candidate_writer(
                        run_id=self.run_id,
                        plan_digest=plan.digest,
                        source_revision=plan.source_identity.revision,
                        locator=candidate_locator,
                        payload=payload,
                        payload_digest=manifest.digest,
                        byte_size=manifest.byte_size,
                    )
                    if (
                        remote_receipt.run_id != self.run_id
                        or remote_receipt.plan_digest != plan.digest
                        or remote_receipt.source_revision != plan.source_identity.revision
                        or remote_receipt.locator != candidate_locator
                        or remote_receipt.payload_digest != manifest.digest
                        or remote_receipt.byte_size != manifest.byte_size
                    ):
                        raise CacheBlobMigrationEvidenceMismatchError(
                            "S3 migration candidate receipt does not bind this exact plan",
                            context={"operation": "migration.stage", "run_id": self.run_id},
                        )
                    written_locator = remote_receipt.locator
                else:
                    written_locator = self.destination.payload_backend.write_blob(candidate_id, payload)
                expected_written_locator = candidate_locator
                if self.destination.topology.qualified_profile.pair == ("memory", "memory"):
                    expected_written_locator = f"memory://{candidate_locator}"
                if written_locator != expected_written_locator:
                    raise CacheBlobMigrationEvidenceMismatchError(
                        "candidate backend returned an unexpected locator",
                        context={"operation": "migration.stage", "run_id": self.run_id},
                    )
                candidate_manifest = sign_current_manifest(
                    replace(manifest, locator=candidate_locator), self._evidence_key
                )
                candidate_raw = candidate_manifest.canonical_bytes()
                self._authenticated_manifest(candidate_raw, role="candidate")
                candidate = AuthorityInventoryEntry(
                    key=snapshot.key,
                    generation=snapshot.generation,
                    locator=candidate_locator,
                    manifest=candidate_raw,
                    payload_digest=manifest.digest,
                    byte_size=manifest.byte_size,
                )
                candidates.append(candidate)
                receipt = self._candidate_receipt(plan, tuple(candidates))
                self._destination_authority.record_verified_candidate(
                    receipt=receipt, entries=tuple(candidates)
                )
                batch = self._batch_receipt(plan, candidate, len(candidates) - 1)
                batch_references.append(batch.candidate_digest)
                current = self._write_evidence(
                    self._new_evidence(
                        state=MaintenanceEvidenceState.STAGING,
                        plan_digest=plan.digest,
                        source_identity=plan.source_identity,
                        destination_identity=plan.destination_identity,
                        completed_steps=("inspect", "plan"),
                        candidate_batch_references=tuple(batch_references),
                        candidate_checkpoint_revision=plan.destination_identity.revision,
                        candidate_entry_count=len(candidates),
                        candidate_byte_count=sum(entry.byte_size for entry in candidates),
                        completed_output_digests={"plan": plan.digest},
                    )
                )
            if page.exhausted:
                break
            cursor = page.next_cursor
        if seen_keys != set(expected):
            raise ValueError("source inventory changed; reinspection is required")
        self._revalidate_identities(plan)
        self._candidate_entries = tuple(candidates)
        receipt = self._candidate_receipt(plan, self._candidate_entries)
        if receipt.entry_count != len(plan.assessments):
            raise CacheBlobMigrationEvidenceMismatchError(
                "authority candidate does not cover every planned source entry",
                context={"operation": "migration.stage", "run_id": self.run_id},
            )
        self._destination_authority.record_verified_candidate(
            receipt=receipt, entries=self._candidate_entries
        )
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.STAGED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=("inspect", "plan", "stage"),
                candidate_receipt=receipt,
                candidate_batch_references=tuple(batch_references),
                candidate_checkpoint_revision=plan.destination_identity.revision,
                candidate_entry_count=receipt.entry_count,
                candidate_byte_count=receipt.byte_count,
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
        """Recover only descriptors durably attributed by the authority."""
        receipt = evidence.candidate_receipt
        if receipt is None:
            raise CacheBlobMigrationEvidenceMismatchError(
                "candidate receipt is required for resume",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        candidates = self._authority_candidate_entries(plan)
        if (
            receipt.plan_digest != plan.digest
            or receipt.candidate_digest != candidate_digest(candidates)
            or receipt.entry_count != len(candidates)
            or receipt.byte_count != sum(item.byte_size for item in candidates)
            or evidence.candidate_entry_count != len(candidates)
            or evidence.candidate_byte_count != sum(item.byte_size for item in candidates)
            or evidence.candidate_checkpoint_revision != plan.destination_identity.revision
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "candidate descriptors do not match authenticated evidence",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        batches = tuple(
            self._batch_receipt(plan, entry, ordinal)
            for ordinal, entry in enumerate(candidates)
        )
        if tuple(batch.candidate_digest for batch in batches) != evidence.candidate_batch_references:
            raise CacheBlobMigrationEvidenceMismatchError(
                "authority candidate batches do not match authenticated evidence",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        self._verify_candidate_outputs(candidates)
        return candidates

    def _attributed_staging_candidates(
        self, evidence: MaintenanceRunEvidence, plan: MigrationPlan
    ) -> tuple[AuthorityInventoryEntry, ...]:
        """Load only exact candidate effects attributable to a STAGING run.

        Candidate payload presence is deliberately not consulted.  An empty
        authority result is valid only for evidence that records no completed
        candidate batch, which keeps the pre-checkpoint orphan boundary
        outside resume and abort authority.
        """
        candidates = self._authority_candidate_entries(plan)
        if not candidates:
            if (
                evidence.candidate_batch_references
                or evidence.candidate_checkpoint_revision is not None
                or evidence.candidate_entry_count
                or evidence.candidate_byte_count
                or evidence.cleanup_debt
            ):
                raise CacheBlobMigrationEvidenceMismatchError(
                    "staging evidence claims candidate effects absent from authority evidence",
                    context={"operation": "migration.abort", "run_id": self.run_id},
                )
            return ()
        batches = tuple(
            self._batch_receipt(plan, entry, ordinal)
            for ordinal, entry in enumerate(candidates)
        )
        if (
            evidence.candidate_checkpoint_revision != plan.destination_identity.revision
            or evidence.candidate_entry_count != len(candidates)
            or evidence.candidate_byte_count != sum(entry.byte_size for entry in candidates)
            or evidence.candidate_batch_references
            != tuple(batch.candidate_digest for batch in batches)
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "staging candidate batches do not match authority evidence",
                context={"operation": "migration.abort", "run_id": self.run_id},
            )
        return candidates

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
                    **self._candidate_progress(evidence),
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
                **self._candidate_progress(evidence),
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
        record_candidate = getattr(
            self._destination_authority, "record_verified_candidate", None
        )
        if callable(record_candidate):
            record_candidate(receipt=receipt, entries=candidates)
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
                **self._candidate_progress(evidence),
                activation_receipt=activation,
                authority_receipts=(activation_digest,),
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "activate": activation_digest,
                },
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.ACTIVATED, True, self.evidence_path)

    def _activated_evidence(self, plan: MigrationPlan) -> MaintenanceRunEvidence:
        """Return the one authenticated activation still eligible for offline action."""
        if not isinstance(plan, MigrationPlan) or plan.run_id != self.run_id:
            raise CacheBlobMigrationEvidenceMismatchError(
                "migration plan does not belong to this explicit run",
                context={"operation": "migration.activation", "run_id": self.run_id},
            )
        evidence = self.read_evidence()
        if evidence.state is not MaintenanceEvidenceState.ACTIVATED:
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "rollback or finalize requires the activated offline migration state",
                context={"operation": "migration.activation", "run_id": self.run_id},
            )
        if evidence.plan_digest != plan.digest or evidence.activation_receipt is None:
            raise CacheBlobMigrationEvidenceMismatchError(
                "activated maintenance evidence does not bind the supplied plan",
                context={"operation": "migration.activation", "run_id": self.run_id},
            )
        self._revalidate_activated_receipt(evidence, plan)
        return evidence

    @staticmethod
    def _confirmation_digest(*, action: str, evidence: MaintenanceRunEvidence) -> str:
        """Bind an explicit operator action to one authenticated activation receipt."""
        receipt = evidence.activation_receipt
        candidate = evidence.candidate_receipt
        if receipt is None or candidate is None:
            raise CacheBlobMigrationEvidenceMismatchError(
                "offline action requires a complete activated authority receipt",
                context={"operation": f"migration.{action}"},
            )
        material = json.dumps(
            {
                "action": action,
                "activation_revision": receipt.activation_revision,
                "candidate_digest": candidate.candidate_digest,
                "plan_digest": evidence.plan_digest,
                "prior_entry_count": (
                    0 if receipt.prior_store is None else receipt.prior_store.entry_count
                ),
                "prior_revision": 0 if receipt.prior_store is None else receipt.prior_store.revision,
                "run_id": evidence.run_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def finalize_confirmation(self, plan: MigrationPlan) -> str:
        """Return the exact D-15 confirmation required to seal one activation."""
        evidence = self._activated_evidence(plan)
        return self._confirmation_digest(action="finalize", evidence=evidence)

    def rollback(self, plan: MigrationPlan) -> RollbackReceipt:
        """Restore the retained prior selection while the activation remains offline."""
        evidence = self._activated_evidence(plan)
        rollback = self._destination_authority.rollback_verified_candidate(run_id=self.run_id)
        rollback_digest = hashlib.sha256(
            f"{rollback.rollback_revision}:{self.run_id}".encode("ascii")
        ).hexdigest()
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.ROLLED_BACK,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=("inspect", "plan", "stage", "verify", "activate", "rollback"),
                candidate_receipt=evidence.candidate_receipt,
                **self._candidate_progress(evidence),
                activation_receipt=evidence.activation_receipt,
                authority_receipts=(*evidence.authority_receipts, rollback_digest),
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "rollback": rollback_digest,
                },
            )
        )
        return rollback

    def finalize(self, plan: MigrationPlan, *, confirmation: str) -> FinalizeReceipt:
        """Seal rollback eligibility after an explicit D-15-bound confirmation."""
        evidence = self.read_evidence()
        if evidence.state is MaintenanceEvidenceState.FINALIZED:
            expected_confirmation = self._confirmation_digest(action="finalize", evidence=evidence)
            if not isinstance(confirmation, str) or not hmac.compare_digest(
                confirmation, expected_confirmation
            ):
                raise CacheBlobMigrationOfflineDecisionRequiredError(
                    "finalize requires the exact explicit confirmation",
                    context={"operation": "migration.finalize", "run_id": self.run_id},
                )
            if evidence.plan_digest != plan.digest or evidence.activation_receipt is None:
                raise CacheBlobMigrationEvidenceMismatchError(
                    "finalized maintenance evidence does not bind the supplied plan",
                    context={"operation": "migration.finalize", "run_id": self.run_id},
                )
            if self._destination_authority.publication_state() is not AuthorityPublicationState.ACTIVE:
                raise CacheBlobMigrationEvidenceMismatchError(
                    "finalized evidence no longer matches the authority state",
                    context={"operation": "migration.finalize", "run_id": self.run_id},
                )
            return FinalizeReceipt(
                run_id=self.run_id,
                finalized_revision=evidence.activation_receipt.activation_revision,
            )

        evidence = self._activated_evidence(plan)
        expected_confirmation = self._confirmation_digest(action="finalize", evidence=evidence)
        if not isinstance(confirmation, str) or not hmac.compare_digest(
            confirmation, expected_confirmation
        ):
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "finalize requires the exact explicit confirmation",
                context={"operation": "migration.finalize", "run_id": self.run_id},
            )
        finalized = self._destination_authority.finalize_verified_candidate(run_id=self.run_id)
        finalized_digest = hashlib.sha256(
            f"{finalized.finalized_revision}:{expected_confirmation}".encode("ascii")
        ).hexdigest()
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.FINALIZED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=("inspect", "plan", "stage", "verify", "activate", "finalize"),
                candidate_receipt=evidence.candidate_receipt,
                **self._candidate_progress(evidence),
                activation_receipt=evidence.activation_receipt,
                authority_receipts=(*evidence.authority_receipts, finalized_digest),
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "finalize": finalized_digest,
                },
            )
        )
        return finalized

    def _retained_prior_entries(self) -> tuple[AuthorityInventoryEntry, ...]:
        """Load exact authenticated retained-prior entries from the authority."""
        reader = getattr(self._destination_authority, "retained_prior_entries", None)
        if not callable(reader):
            raise CacheBlobMigrationEvidenceMismatchError(
                "destination authority cannot expose retained migration entries",
                context={"operation": "migration.purge", "run_id": self.run_id},
            )
        snapshots = reader(run_id=self.run_id)
        if not isinstance(snapshots, tuple):
            raise CacheBlobMigrationEvidenceMismatchError(
                "destination authority returned invalid retained migration entries",
                context={"operation": "migration.purge", "run_id": self.run_id},
            )
        entries: list[AuthorityInventoryEntry] = []
        for snapshot in snapshots:
            try:
                manifest = self._authenticated_manifest(snapshot.manifest, role="retained prior")
            except (AttributeError, TypeError, ValueError) as exc:
                raise CacheBlobMigrationEvidenceMismatchError(
                    "retained prior entry cannot be authenticated",
                    context={"operation": "migration.purge", "run_id": self.run_id},
                ) from exc
            if (
                manifest.key != snapshot.key
                or manifest.generation != snapshot.generation
                or manifest.locator != snapshot.locator
            ):
                raise CacheBlobMigrationEvidenceMismatchError(
                    "retained prior entry does not match its authenticated manifest",
                    context={"operation": "migration.purge", "run_id": self.run_id},
                )
            entries.append(
                AuthorityInventoryEntry(
                    key=snapshot.key,
                    generation=snapshot.generation,
                    locator=snapshot.locator,
                    manifest=bytes(snapshot.manifest),
                    payload_digest=manifest.digest,
                    byte_size=manifest.byte_size,
                )
            )
        if len({(entry.key, entry.generation) for entry in entries}) != len(entries):
            raise CacheBlobMigrationEvidenceMismatchError(
                "retained prior entries contain duplicate identities",
                context={"operation": "migration.purge", "run_id": self.run_id},
            )
        return tuple(sorted(entries, key=lambda entry: (entry.key, entry.generation)))

    def _finalized_evidence(self, plan: MigrationPlan) -> MaintenanceRunEvidence:
        """Require signed finalized evidence and a still-matching source revision."""
        if not isinstance(plan, MigrationPlan) or plan.run_id != self.run_id:
            raise CacheBlobMigrationEvidenceMismatchError(
                "migration plan does not belong to this explicit run",
                context={"operation": "migration.purge", "run_id": self.run_id},
            )
        evidence = self.read_evidence()
        if evidence.state not in {
            MaintenanceEvidenceState.FINALIZED,
            MaintenanceEvidenceState.PURGE_PENDING,
            MaintenanceEvidenceState.PURGED,
        }:
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "purge requires the separately finalized offline migration state",
                context={"operation": "migration.purge", "run_id": self.run_id},
            )
        source_identity = self._source_authority.identity_snapshot()
        if (
            evidence.plan_digest != plan.digest
            or evidence.activation_receipt is None
            or evidence.acknowledgement
            != self._acknowledgement(plan.digest, source_identity)
            or source_identity != plan.source_identity
        ):
            raise CacheBlobMigrationPlanStaleError(
                "purge requires the exact stopped-worker source revision",
                context={"operation": "migration.purge", "run_id": self.run_id},
            )
        if self._destination_authority.publication_state() is not AuthorityPublicationState.ACTIVE:
            raise CacheBlobMigrationEvidenceMismatchError(
                "purge requires the authority's finalized active selection",
                context={"operation": "migration.purge", "run_id": self.run_id},
            )
        return evidence

    @staticmethod
    def _retirement_digest(entry: AuthorityInventoryEntry) -> str:
        """Name one exact external cleanup obligation without exposing payload bytes."""
        material = "\x00".join(
            (entry.key, entry.generation, entry.locator, entry.manifest_digest)
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def _purge_confirmation_digest(
        self,
        evidence: MaintenanceRunEvidence,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> str:
        """Bind destructive confirmation to every retained prior locator and byte count."""
        material = json.dumps(
            {
                "base_confirmation": self._confirmation_digest(action="purge", evidence=evidence),
                "prior_byte_count": sum(entry.byte_size for entry in entries),
                "prior_digest": candidate_digest(entries),
                "prior_entry_count": len(entries),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def abort(self, plan: MigrationPlan) -> AbortReceipt:
        """Retire only this run's authenticated, still-unactivated candidate bytes."""
        evidence = self.read_evidence()
        if evidence.state in {
            MaintenanceEvidenceState.ACTIVATED,
            MaintenanceEvidenceState.FINALIZED,
            MaintenanceEvidenceState.PURGE_PENDING,
            MaintenanceEvidenceState.PURGED,
            MaintenanceEvidenceState.ROLLED_BACK,
        }:
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "abort is unavailable after activation; use rollback or finalize",
                context={"operation": "migration.abort", "run_id": self.run_id},
            )
        if evidence.state is MaintenanceEvidenceState.ABORTED:
            if evidence.plan_digest != plan.digest:
                raise CacheBlobMigrationEvidenceMismatchError(
                    "aborted maintenance evidence does not bind the supplied plan",
                    context={"operation": "migration.abort", "run_id": self.run_id},
                )
            deleted_entries = evidence.candidate_entry_count
            return AbortReceipt(run_id=self.run_id, deleted_entries=deleted_entries)
        if evidence.state not in {
            MaintenanceEvidenceState.INSPECTED,
            MaintenanceEvidenceState.PLANNED,
            MaintenanceEvidenceState.STAGING,
            MaintenanceEvidenceState.STAGED,
            MaintenanceEvidenceState.VERIFYING,
            MaintenanceEvidenceState.VERIFIED,
        }:
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "abort requires recorded unactivated candidate evidence",
                context={"operation": "migration.abort", "run_id": self.run_id},
            )
        self._validate_plan(plan)
        evidence = self._expect_evidence(evidence.state, plan_digest=plan.digest)
        candidates = (
            self._attributed_staging_candidates(evidence, plan)
            if evidence.state is MaintenanceEvidenceState.STAGING
            else self._candidate_from_evidence(evidence, plan)
        )
        cleanup_debt: list[str] = []
        for candidate in candidates:
            retirement = self._retirement_digest(candidate)
            try:
                manifest = self._authenticated_manifest(candidate.manifest, role="candidate")
                destination_io = self.destination._materialize_authority_store()
                try:
                    with destination_io.open_snapshot(candidate.locator, {}) as snapshot:
                        payload = snapshot.path.read_bytes()
                except FileNotFoundError:
                    continue
                except OSError:
                    cleanup_debt.append(retirement)
                    continue
                if (
                    manifest.key != candidate.key
                    or manifest.generation != candidate.generation
                    or manifest.locator != candidate.locator
                    or manifest.digest != candidate.payload_digest
                    or manifest.byte_size != candidate.byte_size
                    or hashlib.sha256(payload).hexdigest() != candidate.payload_digest
                    or len(payload) != candidate.byte_size
                ):
                    cleanup_debt.append(retirement)
                    continue
                self.destination.delete_migration_payload(candidate.locator)
            except (OSError, ValueError):
                cleanup_debt.append(retirement)
        cleanup_debt = list(dict.fromkeys(cleanup_debt))
        if cleanup_debt:
            self._write_evidence(
                self._new_evidence(
                    state=MaintenanceEvidenceState.STAGING,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=evidence.completed_steps,
                    candidate_receipt=evidence.candidate_receipt,
                    **self._candidate_progress(evidence),
                    activation_receipt=evidence.activation_receipt,
                    authority_receipts=evidence.authority_receipts,
                    cleanup_debt=tuple(cleanup_debt),
                    completed_output_digests=evidence.completed_output_digests,
                )
            )
            return AbortReceipt(
                run_id=self.run_id,
                deleted_entries=len(candidates),
                state=MaintenanceEvidenceState.STAGING,
            )
        if candidates:
            receipt = self._candidate_receipt(plan, candidates)
            discard = getattr(self._destination_authority, "discard_verified_candidate", None)
            if not callable(discard):
                raise CacheBlobMigrationEvidenceMismatchError(
                    "destination authority cannot retire an unactivated candidate",
                    context={"operation": "migration.abort", "run_id": self.run_id},
                )
            discard(receipt=receipt, entries=candidates)
        abort_digest = hashlib.sha256(
            f"{self.run_id}:{len(candidates)}".encode("utf-8")
        ).hexdigest()
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.ABORTED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=(*evidence.completed_steps, "abort"),
                candidate_receipt=evidence.candidate_receipt,
                **self._candidate_progress(evidence),
                activation_receipt=evidence.activation_receipt,
                authority_receipts=evidence.authority_receipts,
                cleanup_debt=(),
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "abort": abort_digest,
                },
            )
        )
        return AbortReceipt(run_id=self.run_id, deleted_entries=len(candidates))

    def purge_confirmation(self, plan: MigrationPlan) -> str:
        """Return the exact separately confirmed D-15 retirement action token."""
        evidence = self._finalized_evidence(plan)
        return self._purge_confirmation_digest(evidence, self._retained_prior_entries())

    def purge(self, plan: MigrationPlan, *, confirmation: str) -> PurgeReceipt:
        """Apply idempotent retained-prior cleanup without revising activation state."""
        evidence = self._finalized_evidence(plan)
        entries = self._retained_prior_entries()
        expected_confirmation = self._purge_confirmation_digest(evidence, entries)
        if not isinstance(confirmation, str) or not hmac.compare_digest(
            confirmation, expected_confirmation
        ):
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "purge requires the exact separately confirmed confirmation for the prior selection",
                context={"operation": "migration.purge", "run_id": self.run_id},
            )
        if evidence.state is MaintenanceEvidenceState.PURGED:
            return PurgeReceipt(
                run_id=self.run_id,
                purged_entries=len(entries),
                pending_entries=0,
                cleanup_debt=(),
                completed=True,
            )
        if evidence.state is MaintenanceEvidenceState.FINALIZED:
            self._write_evidence(
                self._new_evidence(
                    state=MaintenanceEvidenceState.PURGE_PENDING,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=evidence.completed_steps,
                    candidate_receipt=evidence.candidate_receipt,
                    **self._candidate_progress(evidence),
                    activation_receipt=evidence.activation_receipt,
                    authority_receipts=evidence.authority_receipts,
                    completed_output_digests=evidence.completed_output_digests,
                )
            )
            evidence = self.read_evidence()

        pending: list[str] = []
        for entry in entries:
            try:
                self.destination.delete_migration_payload(entry.locator)
            except Exception:
                pending.append(self._retirement_digest(entry))
        if pending:
            self._write_evidence(
                self._new_evidence(
                    state=MaintenanceEvidenceState.PURGE_PENDING,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=evidence.completed_steps,
                    candidate_receipt=evidence.candidate_receipt,
                    **self._candidate_progress(evidence),
                    activation_receipt=evidence.activation_receipt,
                    authority_receipts=evidence.authority_receipts,
                    cleanup_debt=tuple(pending),
                    completed_output_digests=evidence.completed_output_digests,
                )
            )
            return PurgeReceipt(
                run_id=self.run_id,
                purged_entries=len(entries) - len(pending),
                pending_entries=len(pending),
                cleanup_debt=tuple(pending),
                completed=False,
            )
        purge_digest = hashlib.sha256(
            "\x00".join(self._retirement_digest(entry) for entry in entries).encode("utf-8")
        ).hexdigest()
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.PURGED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=(*evidence.completed_steps, "purge"),
                candidate_receipt=evidence.candidate_receipt,
                **self._candidate_progress(evidence),
                activation_receipt=evidence.activation_receipt,
                authority_receipts=evidence.authority_receipts,
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "purge": purge_digest,
                },
            )
        )
        return PurgeReceipt(
            run_id=self.run_id,
            purged_entries=len(entries),
            pending_entries=0,
            cleanup_debt=(),
            completed=True,
        )

    def rebuild_projection(
        self,
        plan: MigrationPlan,
        controller: ProjectionController,
    ) -> ProjectionResult:
        """Run an explicit derived rebuild only after canonical acceptance.

        Projection publication is an external derived effect. Its failure is
        reported beside immutable offline-maintenance evidence and can never
        revoke, select, or otherwise redefine canonical authority state.
        """
        if not isinstance(plan, MigrationPlan) or plan.run_id != self.run_id:
            raise CacheBlobMigrationEvidenceMismatchError(
                "migration plan does not belong to this explicit run",
                context={"operation": "migration.projection", "run_id": self.run_id},
            )
        if not isinstance(controller, ProjectionController):
            raise TypeError("projection rebuild requires a ProjectionController")
        if plan.plan_kind is MigrationPlanKind.REBUILD:
            evidence = self._expect_rebuild_evidence(
                MaintenanceEvidenceState.REBUILD_ACCEPTED, plan=plan
            )
            receipt = evidence.authority_receipts[-1] if evidence.authority_receipts else None
        else:
            evidence = self.read_evidence()
            if evidence.state is not MaintenanceEvidenceState.ACTIVATED or (
                evidence.plan_digest != plan.digest or evidence.activation_receipt is None
            ):
                raise CacheBlobMigrationOfflineDecisionRequiredError(
                    "projection rebuild requires confirmed offline canonical acceptance",
                    context={"operation": "migration.projection", "run_id": self.run_id},
                )
            self._revalidate_activated_receipt(evidence, plan)
            receipt = evidence.activation_receipt
        try:
            result = controller.rebuild(requested="offline")
        except Exception as error:
            checkpoint = getattr(controller, "_active_checkpoint", None)
            outcome = ProjectionOutcome(
                controller.projection_name,
                ProjectionStatus.DIRTY,
                checkpoint,
                type(error).__name__,
            )
            return ProjectionResult(
                False,
                checkpoint,
                0,
                receipt=receipt,
                projection_status=ProjectionStatus.DIRTY.value,
                outcome=outcome,
            )
        outcome = ProjectionOutcome(
            controller.projection_name,
            ProjectionStatus.CURRENT,
            result.checkpoint,
        )
        return replace(
            result,
            receipt=receipt,
            projection_status=ProjectionStatus.CURRENT.value,
            outcome=outcome,
        )

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
        if plan.plan_kind is MigrationPlanKind.REBUILD:
            self._validate_rebuild_plan(plan)
            if evidence.state is MaintenanceEvidenceState.REBUILDING:
                self._validate_rebuild_receipts(plan, evidence, require_complete=False)
                return self.stage_rebuild(plan)
            if evidence.state is MaintenanceEvidenceState.REBUILD_STAGED:
                self._validate_rebuild_receipts(plan, evidence, require_complete=True)
                return self.verify_rebuild(plan)
            if evidence.state is MaintenanceEvidenceState.REBUILD_VERIFYING:
                self._validate_rebuild_receipts(plan, evidence, require_complete=True)
                return self.verify_rebuild(plan)
            if evidence.state is MaintenanceEvidenceState.REBUILD_VERIFIED:
                self._validate_rebuild_receipts(plan, evidence, require_complete=True)
                return MigrationStepResult(
                    MaintenanceEvidenceState.REBUILD_VERIFIED, True, self.evidence_path
                )
            if evidence.state is MaintenanceEvidenceState.REBUILD_ACCEPTED:
                self._validate_rebuild_receipts(plan, evidence, require_complete=True)
                return MigrationStepResult(
                    MaintenanceEvidenceState.REBUILD_ACCEPTED, True, self.evidence_path
                )
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "rebuild resume state requires an explicit offline action",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        if evidence.state is MaintenanceEvidenceState.ACTIVATED:
            self._revalidate_activated_receipt(evidence, plan)
            return MigrationStepResult(MaintenanceEvidenceState.ACTIVATED, True, self.evidence_path)
        if evidence.state is MaintenanceEvidenceState.VERIFIED:
            receipt = evidence.candidate_receipt
            assert receipt is not None
            activation_reader = getattr(
                self._destination_authority, "activation_receipt_for_candidate", None
            )
            activation = activation_reader(receipt) if callable(activation_reader) else None
            if activation is not None:
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
                        **self._candidate_progress(evidence),
                        activation_receipt=activation,
                        authority_receipts=(activation_digest,),
                        completed_output_digests={
                            **evidence.completed_output_digests,
                            "activate": activation_digest,
                        },
                    )
                )
                self._revalidate_activated_receipt(self.read_evidence(), plan)
                return MigrationStepResult(
                    MaintenanceEvidenceState.ACTIVATED, True, self.evidence_path
                )
            self._candidate_from_evidence(evidence, plan)
            self._validate_plan(plan)
            return MigrationStepResult(MaintenanceEvidenceState.VERIFIED, True, self.evidence_path)
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
    "AbortReceipt",
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
    "PurgeReceipt",
    "ReleaseWindow",
    "VersionEdge",
    "inspect_migration_store",
    "inspect_store_path",
    "render_migration_report",
]
