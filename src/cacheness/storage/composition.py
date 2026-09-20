"""Typed, role-aware construction for direct :class:`BlobStore` instances.

The composition layer deliberately selects participants only.  It does not add
another lifecycle coordinator, lock, queue, or transaction boundary: the
selected ``LifecycleAuthority`` remains the sole lifecycle authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .lifecycle_authority import LifecycleAuthority
from .memory_lifecycle_authority import InMemoryLifecycleAuthority
from .obstore_generation_io import ObstoreGenerationIO
from .sqlite_lifecycle_authority import SqliteLifecycleAuthority
from .transport_evidence import PayloadTransportObservation


class CompositionValidationError(ValueError):
    """Raised when a topology cannot be resolved without ambiguity."""


class CapabilityRequirementError(CompositionValidationError):
    """Raised before lifecycle I/O when a topology misses a required guarantee."""


@dataclass(frozen=True)
class TopologyQualificationRequirements:
    """Immutable release contract for one explicitly supported topology.

    This declaration contains no observed readiness or latest-run state.
    Real-service observations belong in sanitized release evidence outside
    runtime composition.
    """

    coordination_scope: str
    durability_atomicity_boundary: str
    progress_outcomes: frozenset[str]
    service_prerequisites: tuple[str, ...]
    projection_available: bool
    evidence_requirement_id: str
    evidence_schema_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.coordination_scope, str) or not self.coordination_scope:
            raise CompositionValidationError(
                "coordination_scope must be a non-empty string"
            )
        if (
            not isinstance(self.durability_atomicity_boundary, str)
            or not self.durability_atomicity_boundary
        ):
            raise CompositionValidationError(
                "durability_atomicity_boundary must be a non-empty string"
            )
        if (
            not isinstance(self.progress_outcomes, frozenset)
            or not self.progress_outcomes
        ):
            raise CompositionValidationError(
                "progress_outcomes must be a non-empty frozenset"
            )
        if not all(
            isinstance(outcome, str) and outcome for outcome in self.progress_outcomes
        ):
            raise CompositionValidationError(
                "progress_outcomes must contain non-empty strings"
            )
        if not isinstance(self.service_prerequisites, tuple) or not all(
            isinstance(prerequisite, str) and prerequisite
            for prerequisite in self.service_prerequisites
        ):
            raise CompositionValidationError(
                "service_prerequisites must be a tuple of non-empty strings"
            )
        if type(self.projection_available) is not bool:
            raise CompositionValidationError("projection_available must be a bool")
        for name in ("evidence_requirement_id", "evidence_schema_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise CompositionValidationError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class QualifiedTopologyProfile:
    """One exact authority/payload pairing with its immutable qualification contract."""

    authority_identity: str
    payload_identity: str
    requirements: TopologyQualificationRequirements

    def __post_init__(self) -> None:
        for name in ("authority_identity", "payload_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise CompositionValidationError(f"{name} must be a non-empty string")
        if not isinstance(self.requirements, TopologyQualificationRequirements):
            raise CompositionValidationError(
                "requirements must be TopologyQualificationRequirements"
            )

    @property
    def pair(self) -> tuple[str, str]:
        """Return the normalized authority/payload lookup key."""
        return (self.authority_identity, self.payload_identity)


def _qualified_profile(
    authority_identity: str,
    payload_identity: str,
    *,
    coordination_scope: str,
    durability_atomicity_boundary: str,
    progress_outcomes: frozenset[str],
    service_prerequisites: tuple[str, ...],
    evidence_requirement_id: str,
    evidence_schema_id: str,
) -> QualifiedTopologyProfile:
    """Build one catalog row while keeping the catalog declaration concise."""
    return QualifiedTopologyProfile(
        authority_identity=authority_identity,
        payload_identity=payload_identity,
        requirements=TopologyQualificationRequirements(
            coordination_scope=coordination_scope,
            durability_atomicity_boundary=durability_atomicity_boundary,
            progress_outcomes=progress_outcomes,
            service_prerequisites=service_prerequisites,
            projection_available=True,
            evidence_requirement_id=evidence_requirement_id,
            evidence_schema_id=evidence_schema_id,
        ),
    )


BUILTIN_QUALIFIED_TOPOLOGY_PROFILES: Mapping[
    tuple[str, str], QualifiedTopologyProfile
] = MappingProxyType(
    {
        ("memory", "memory"): _qualified_profile(
            "memory",
            "memory",
            coordination_scope="one_process",
            durability_atomicity_boundary=(
                "atomic only within one process; no crash durability"
            ),
            progress_outcomes=frozenset({"success", "conflict"}),
            service_prerequisites=("no external service",),
            evidence_requirement_id="local-memory-contract",
            evidence_schema_id="phase5-local-contract-v1",
        ),
        ("sqlite", "filesystem"): _qualified_profile(
            "sqlite",
            "filesystem",
            coordination_scope="one_host_multiple_processes",
            durability_atomicity_boundary=(
                "SQLite transaction is authoritative; immutable filesystem "
                "generations reconcile outside cross-resource ACID"
            ),
            progress_outcomes=frozenset({"success", "conflict", "retryable_timeout"}),
            service_prerequisites=(
                "explicit initialization before shared workers",
                "writable local filesystem",
            ),
            evidence_requirement_id="local-sqlite-filesystem-contract",
            evidence_schema_id="phase5-local-contract-v1",
        ),
        ("postgresql", "s3"): _qualified_profile(
            "postgresql",
            "s3",
            coordination_scope="multiple_hosts",
            durability_atomicity_boundary=(
                "PostgreSQL transaction is authoritative; immutable Amazon S3 "
                "objects reconcile outside cross-resource ACID"
            ),
            progress_outcomes=frozenset(
                {
                    "success",
                    "conflict",
                    "retryable_serialization",
                    "retryable_deadlock",
                    "retryable_lock_timeout",
                    "retryable_statement_timeout",
                    "retryable_connection_timeout",
                }
            ),
            service_prerequisites=(
                "explicit PostgreSQL initialization before shared workers",
                "real PostgreSQL service",
                "real Amazon S3 bucket and test-owned prefix",
                "shared external manifest signing key",
            ),
            evidence_requirement_id="live-postgresql-amazon-s3",
            evidence_schema_id="phase5-live-service-evidence-v1",
        ),
    }
)


def qualified_topology_profiles() -> Mapping[tuple[str, str], QualifiedTopologyProfile]:
    """Return the immutable built-in qualification catalog.

    Registration answers whether a participant can be constructed. This
    catalog answers the separate question of whether a pairing is supported.
    """
    return BUILTIN_QUALIFIED_TOPOLOGY_PROFILES


class BackendRole(str, Enum):
    """The only participant roles accepted by a store topology."""

    PAYLOAD = "payload"
    AUTHORITY = "authority"
    PROJECTION = "projection"


# ``BackendRole.PROJECTION`` is intentionally the one vocabulary for every
# derived consumer. A projection may copy committed catalog state, but it
# cannot become a second lifecycle authority by choosing a different role.
ProjectionRole = BackendRole.PROJECTION


@runtime_checkable
class ProjectionSink(Protocol):
    """A derived-only destination for bounded committed catalog batches.

    Projection sinks own their own checkpoints and idempotent apply behavior.
    They never receive lifecycle mutation primitives, payload locators for
    cleanup, or authority credentials. The controller accepts the legacy
    method spellings while this protocol names the clean Phase 4 surface.
    """

    projection_name: str

    def apply_projection_batch(self, batch: object) -> None:
        """Apply one idempotently identified derived batch."""

    def save_projection_checkpoint(self, checkpoint: object) -> None:
        """Persist progress only after a successful batch apply."""

    def load_projection_checkpoint(self) -> object | None:
        """Return the last committed derived checkpoint, if any."""


@runtime_checkable
class PayloadGenerationIOProvider(Protocol):
    """Materialize the one guarded generation-I/O primitive a store consumes.

    This payload role deliberately supplies only staging, snapshot, publication,
    and deletion mechanics. It does not receive authority mutation methods or
    decide lifecycle sequencing, recovery, or catalog completeness.
    """

    def materialize_handler_io(self) -> object:
        """Return the participant-rooted guarded generation-I/O primitive."""
        ...


@runtime_checkable
class PayloadTransportObservationProvider(Protocol):
    """Optionally observe one validated immutable locator without lifecycle authority.

    The observation is untrusted transport metadata only.  It cannot select a
    generation, mutate authority state, or replace canonical payload hashing.
    """

    def observe_transport(self, locator: str) -> PayloadTransportObservation:
        """Return one exact bounded observation for the supplied locator."""
        ...


class Ownership(str, Enum):
    """The resource owner responsible for one selected participant's close."""

    CALLER = "caller"
    STORE = "store"


_SCOPE_RANK = {"none": 0, "process": 1, "host": 2, "multi_host": 3}


@dataclass(frozen=True)
class ParticipantCapabilities:
    """Truthful semantic capability declaration for one active participant."""

    durable: bool = False
    process_scope: str = "process"
    host_scope: str = "process"
    transaction_scope: str = "none"
    exact_cas: bool = False
    immutable_generations: bool = False
    streaming: bool = False
    listing: bool = False
    portable_query: bool = False
    canonical_scan: bool = False
    index_acceleration: bool = False
    projection_refresh: bool = False
    projection_rebuild: bool = False
    online_rebuild: bool = False
    offline_rebuild: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "durable",
            "exact_cas",
            "immutable_generations",
            "streaming",
            "listing",
            "portable_query",
            "canonical_scan",
            "index_acceleration",
            "projection_refresh",
            "projection_rebuild",
            "online_rebuild",
            "offline_rebuild",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise CompositionValidationError(f"{field_name} must be a bool")
        for field_name in ("process_scope", "host_scope"):
            if getattr(self, field_name) not in _SCOPE_RANK:
                raise CompositionValidationError(
                    f"{field_name} must be one of {sorted(_SCOPE_RANK)}"
                )
        if self.transaction_scope not in {"none", "authority"}:
            raise CompositionValidationError(
                "transaction_scope must be 'none' or 'authority'"
            )

    @classmethod
    def from_participant(
        cls, participant: object, role: str
    ) -> "ParticipantCapabilities":
        """Normalize a participant's local declaration without inferring strength."""
        declared = (
            participant
            if isinstance(participant, Mapping)
            else getattr(participant, "topology_capabilities", None)
        )
        if isinstance(declared, cls):
            return declared
        raw = (
            declared
            if declared is not None
            else getattr(participant, "capabilities", {})
        )
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, Mapping):
            get = raw.get
        else:

            def get(name: str, default: object = None) -> object:
                return getattr(raw, name, default)

        transactional = get("transactional", False)
        return cls(
            durable=_capability_bool(get("durable", False), "durable"),
            process_scope=_capability_scope(
                get("process_scope", "process"), "process_scope"
            ),
            host_scope=_capability_scope(get("host_scope", "process"), "host_scope"),
            transaction_scope=get(
                "transaction_scope",
                "authority"
                if role == BackendRole.AUTHORITY.value and transactional
                else "none",
            ),
            exact_cas=_capability_bool(
                get("exact_cas", get("compare_and_swap", False)), "exact_cas"
            ),
            immutable_generations=_capability_bool(
                get("immutable_generations", False), "immutable_generations"
            ),
            streaming=_capability_bool(get("streaming", False), "streaming"),
            listing=_capability_bool(get("listing", False), "listing"),
            portable_query=_capability_bool(
                get("portable_query", False), "portable_query"
            ),
            canonical_scan=_capability_bool(
                get("canonical_scan", False), "canonical_scan"
            ),
            index_acceleration=_capability_bool(
                get("index_acceleration", False), "index_acceleration"
            ),
            projection_refresh=_capability_bool(
                get("projection_refresh", False), "projection_refresh"
            ),
            projection_rebuild=_capability_bool(
                get("projection_rebuild", False), "projection_rebuild"
            ),
            online_rebuild=_capability_bool(
                get("online_rebuild", False), "online_rebuild"
            ),
            offline_rebuild=_capability_bool(
                get("offline_rebuild", False), "offline_rebuild"
            ),
        )


@dataclass(frozen=True)
class TopologyCapabilities:
    """Conservative capability report derived from the resolved topology."""

    durable: bool
    process_scope: str
    host_scope: str
    transaction_scope: str
    exact_cas: bool
    immutable_generations: bool
    streaming: bool
    listing: bool
    portable_query: bool
    canonical_scan: bool
    index_acceleration: bool
    projection_refresh: bool
    projection_rebuild: bool
    online_rebuild: bool
    offline_rebuild: bool


@dataclass(frozen=True)
class CapabilityMinimum:
    """Immutable caller-required capability values for topology construction."""

    requirements: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.requirements, Mapping):
            raise CompositionValidationError("Capability minimum must be a mapping")
        unsupported = set(self.requirements) - set(
            TopologyCapabilities.__dataclass_fields__
        )
        if unsupported:
            raise CompositionValidationError(
                f"Unsupported capability minimum: {sorted(unsupported)}"
            )
        object.__setattr__(
            self, "requirements", MappingProxyType(dict(self.requirements))
        )

    def require(self, report: TopologyCapabilities) -> None:
        """Fail closed when the active topology is weaker than a requirement."""
        unmet: list[str] = []
        for name, requested in self.requirements.items():
            actual = getattr(report, name)
            if name in {"process_scope", "host_scope"}:
                if not isinstance(requested, str) or requested not in _SCOPE_RANK:
                    raise CompositionValidationError(
                        f"{name} minimum must be a supported scope"
                    )
                if _SCOPE_RANK[actual] < _SCOPE_RANK[requested]:
                    unmet.append(name)
            elif name == "transaction_scope":
                if requested not in {"none", "authority"}:
                    raise CompositionValidationError(
                        "transaction_scope minimum is unsupported"
                    )
                if requested == "authority" and actual != "authority":
                    unmet.append(name)
            elif type(requested) is not bool:
                raise CompositionValidationError(f"{name} minimum must be a bool")
            elif requested and not actual:
                unmet.append(name)
        if unmet:
            raise CapabilityRequirementError(
                "Topology cannot satisfy required capabilities: " + ", ".join(unmet)
            )


@dataclass(frozen=True)
class MetadataRole:
    """Declared lifecycle authority or derived-projection metadata role."""

    kind: str

    def authorizes(self, operation: str) -> bool:
        """Return whether this metadata role can authorize ``operation``."""
        authority_operations = {"query_complete", "promote_catalog"}
        return (
            self.kind == BackendRole.AUTHORITY.value
            and operation in authority_operations
        )


@dataclass(frozen=True)
class BackendRef:
    """One unambiguous name-or-instance participant reference.

    Registered references construct a fresh participant with ``options``.  An
    injected instance never has options merged into it and retains exact object
    identity through topology resolution.
    """

    name: str | None = None
    instance: object | None = None
    options: Mapping[str, object] = field(default_factory=dict)
    ownership: Ownership | str | None = None
    transfer_ownership: bool = False

    def __post_init__(self) -> None:
        has_name = self.name is not None
        has_instance = self.instance is not None
        if has_name == has_instance:
            raise CompositionValidationError(
                "BackendRef requires exactly one of name or instance"
            )
        if has_name and (not isinstance(self.name, str) or not self.name):
            raise CompositionValidationError(
                "BackendRef name must be a non-empty string"
            )
        if has_instance and self.options:
            raise CompositionValidationError(
                "BackendRef options cannot be combined with an injected instance"
            )
        if type(self.transfer_ownership) is not bool:
            raise CompositionValidationError("transfer_ownership must be a bool")
        ownership = self.ownership
        if ownership is not None and not isinstance(ownership, Ownership):
            try:
                ownership = Ownership(ownership)
            except ValueError as error:
                raise CompositionValidationError(
                    "ownership must be caller or store"
                ) from error
        if self.transfer_ownership:
            if ownership not in {None, Ownership.STORE}:
                raise CompositionValidationError(
                    "transfer_ownership conflicts with caller ownership"
                )
            ownership = Ownership.STORE
        if ownership is None:
            ownership = Ownership.CALLER if has_instance else Ownership.STORE
        object.__setattr__(self, "ownership", ownership)
        if not isinstance(self.options, Mapping):
            raise CompositionValidationError("BackendRef options must be a mapping")
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class RoleRegistration:
    """A single named factory declaration for a participant role."""

    role: str
    name: str
    factory: Callable[..., object]
    capabilities: ParticipantCapabilities | None = None

    def construct(self, options: Mapping[str, object]) -> object:
        """Construct one participant through the registered factory."""
        try:
            participant = self.factory(**dict(options))
        except TypeError as error:
            raise CompositionValidationError(
                f"Unable to construct {self.role} participant {self.name!r}"
            ) from error
        if participant is None:
            raise CompositionValidationError(
                f"Registered {self.role} participant {self.name!r} returned None"
            )
        return participant


class RoleRegistry:
    """One registry for built-in and application-defined participant factories."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], RoleRegistration] = {}
        self._register_builtin_participants()

    def _register_builtin_participants(self) -> None:
        # Built-ins take the same path as application registrations. JSON is a
        # derived-only projection; PostgreSQL is classified as derived but not
        # constructible until Phase 5 qualifies a real ProjectionSink.
        self.register(
            BackendRole.PAYLOAD.value,
            "memory",
            _construct_memory_payload,
            capabilities=ObstoreGenerationIO.topology_capabilities,
        )
        self.register(
            BackendRole.PAYLOAD.value,
            "filesystem",
            _construct_filesystem_payload,
            capabilities={
                "durable": True,
                "process_scope": "host",
                "host_scope": "host",
                "immutable_generations": True,
                "streaming": True,
                "listing": True,
            },
        )
        self.register(
            BackendRole.AUTHORITY.value,
            "memory",
            InMemoryLifecycleAuthority,
        )
        self.register(
            BackendRole.AUTHORITY.value,
            "sqlite",
            _construct_sqlite_authority,
            capabilities={
                "durable": True,
                "process_scope": "host",
                "host_scope": "host",
                "transaction_scope": "authority",
                "exact_cas": True,
                "portable_query": True,
                "canonical_scan": True,
                "index_acceleration": False,
            },
        )
        self.register(
            BackendRole.PAYLOAD.value,
            "s3",
            _construct_s3_payload,
            capabilities={
                "durable": True,
                "process_scope": "multi_host",
                "host_scope": "multi_host",
                "immutable_generations": True,
                "streaming": True,
                "listing": True,
            },
        )
        self.register(
            BackendRole.AUTHORITY.value,
            "postgresql",
            _construct_postgresql_authority,
            capabilities={
                "durable": True,
                "process_scope": "multi_host",
                "host_scope": "multi_host",
                "transaction_scope": "authority",
                "exact_cas": True,
                "portable_query": True,
                "canonical_scan": True,
                "index_acceleration": True,
            },
        )
        self.register(
            BackendRole.PROJECTION.value,
            "json",
            _construct_json_projection,
            capabilities={"projection_refresh": True},
        )

    def register(
        self,
        role: str | BackendRole,
        name: str,
        factory: Callable[..., object],
        *,
        capabilities: ParticipantCapabilities | Mapping[str, object] | None = None,
        replace: bool = False,
    ) -> None:
        """Register a factory under exactly one participant role."""
        normalized_role = _normalize_role(role)
        if not isinstance(name, str) or not name:
            raise CompositionValidationError(
                "Participant registration name must be a string"
            )
        if not callable(factory):
            raise CompositionValidationError(
                "Participant registration factory must be callable"
            )
        if capabilities is None:
            capabilities = getattr(factory, "topology_capabilities", None)
        if capabilities is not None and not isinstance(
            capabilities, ParticipantCapabilities
        ):
            capabilities = ParticipantCapabilities.from_participant(
                capabilities, normalized_role
            )
        key = (normalized_role, name)
        if key in self._entries and not replace:
            raise CompositionValidationError(
                f"A {normalized_role} participant named {name!r} is already registered"
            )
        self._entries[key] = RoleRegistration(
            normalized_role,
            name,
            factory,
            capabilities,
        )

    def resolve(self, role: str | BackendRole, name: str) -> RoleRegistration:
        """Resolve a typed role declaration without constructing the participant."""
        normalized_role = _normalize_role(role)
        try:
            return self._entries[(normalized_role, name)]
        except KeyError as error:
            raise CompositionValidationError(
                f"No {normalized_role} participant is registered as {name!r}"
            ) from error

    def construct(
        self,
        role: str | BackendRole,
        name: str,
        options: Mapping[str, object],
    ) -> object:
        """Construct a named participant for the topology that owns it.

        ``StoreTopology`` records a store-owned result before it validates the
        structural role. Keeping that sequence here would leave an invalid
        factory result outside the only close ledger.
        """
        return self.resolve(role, name).construct(options)

    def capabilities(
        self, role: str | BackendRole, name: str
    ) -> ParticipantCapabilities:
        """Return a static declaration suitable for capability preflight."""
        registration = self.resolve(role, name)
        if registration.capabilities is None:
            raise CapabilityRequirementError(
                f"{registration.role} participant {registration.name!r} lacks "
                "a pre-construction capability declaration"
            )
        return registration.capabilities


@dataclass
class ResolvedTopology:
    """Resolved participants plus the ownership ledger used by later lifecycle work."""

    payload: object
    authority: object
    projections: tuple[object, ...]
    capabilities: TopologyCapabilities
    qualified_profile: QualifiedTopologyProfile
    _owned_in_creation_order: tuple[object, ...] = ()
    _closed: bool = False

    def close(self) -> None:
        """Close only owned participants, exactly once, in reverse creation order."""
        if self._closed:
            return
        self._closed = True
        _close_owned_participants(self._owned_in_creation_order)


def _construct_sqlite_authority(*, root: str | Path, **options: object) -> object:
    """Build a local authority only when a topology supplies its explicit root."""
    return SqliteLifecycleAuthority.for_root(Path(root), **options)


def _construct_memory_payload(**options: object) -> object:
    """Construct the guarded in-process obstore participant."""
    return ObstoreGenerationIO.for_memory(**options)


def _construct_filesystem_payload(*, base_dir: str | Path, **options: object) -> object:
    """Construct the guarded LocalStore participant at one managed root."""
    return ObstoreGenerationIO.for_filesystem(base_dir=base_dir, **options)


def _construct_s3_payload(**options: object) -> object:
    """Construct only the guarded Amazon S3 generation-I/O participant.

    The participant uses obstore's standard AWS credential chain. Registration
    remains constructibility, not a live-service qualification claim.
    """
    return ObstoreGenerationIO.for_s3(**options)


def _construct_postgresql_authority(**options: object) -> object:
    """Construct only the narrow PostgreSQL lifecycle authority.

    Importing the storage package never requires psycopg.  Construction keeps
    the authority's actionable missing-extra error and does not initialize or
    migrate a schema.
    """
    from .backends.postgresql_lifecycle_authority import PostgresqlLifecycleAuthority

    participant = PostgresqlLifecycleAuthority(**options)
    participant.qualification_identity = "postgresql"
    return participant


def _construct_json_projection(
    *, metadata_file: str | Path, **options: object
) -> object:
    """Build the JSON carrier only as a derived projection participant."""
    from cacheness.metadata import JsonProjection

    return JsonProjection(Path(metadata_file), **options)


@dataclass(frozen=True)
class StoreTopology:
    """The sole typed composition root for direct BlobStore construction."""

    payload: BackendRef | object
    authority: BackendRef | object
    projections: tuple[BackendRef | object, ...] = ()
    role_registry: RoleRegistry = field(default_factory=RoleRegistry)
    minimum_capabilities: CapabilityMinimum | Mapping[str, object] = field(
        default_factory=CapabilityMinimum
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _as_ref(self.payload))
        object.__setattr__(self, "authority", _as_ref(self.authority))
        projections = tuple(_as_ref(projection) for projection in self.projections)
        projection_identities: set[tuple[str, object]] = set()
        for projection in projections:
            identity = _projection_reference_identity(projection)
            if identity in projection_identities:
                raise CompositionValidationError(
                    "Duplicate projection declaration is not allowed"
                )
            projection_identities.add(identity)
        object.__setattr__(
            self,
            "projections",
            projections,
        )
        if not isinstance(self.role_registry, RoleRegistry):
            raise CompositionValidationError("role_registry must be a RoleRegistry")
        minimum = self.minimum_capabilities
        if not isinstance(minimum, CapabilityMinimum):
            minimum = CapabilityMinimum(minimum)
        object.__setattr__(self, "minimum_capabilities", minimum)

    def resolve(self) -> ResolvedTopology:
        """Resolve each role once and unwind only owned participants on failure."""
        active_registry = self.role_registry
        owned: list[object] = []
        try:
            _record_and_validate_injected_participants(self, owned)
            if self.minimum_capabilities.requirements:
                self.minimum_capabilities.require(
                    _capabilities_from_references(self, active_registry)
                )
            qualified_profile = self.qualification_report()
            payload = _resolve_ref(
                BackendRole.PAYLOAD.value,
                self.payload,
                active_registry,
                owned,
            )
            authority = _resolve_ref(
                BackendRole.AUTHORITY.value,
                self.authority,
                active_registry,
                owned,
            )
            projections = tuple(
                _resolve_ref(
                    BackendRole.PROJECTION.value,
                    projection,
                    active_registry,
                    owned,
                )
                for projection in self.projections
            )
            capabilities = _compose_capabilities(payload, authority, projections)
            self.minimum_capabilities.require(capabilities)
            return ResolvedTopology(
                payload,
                authority,
                projections,
                capabilities,
                qualified_profile,
                tuple(owned),
            )
        except BaseException:
            _close_owned_participants(owned, suppress_errors=True)
            raise

    def capability_report(self) -> TopologyCapabilities:
        """Report constructible capabilities without asserting supported pairing.

        The report avoids participant construction. Use
        :meth:`qualification_report` to inspect the distinct support contract.
        """
        _record_and_validate_injected_participants(self, [])
        return _capabilities_from_references(self, self.role_registry)

    def qualification_report(self) -> QualifiedTopologyProfile:
        """Return one immutable profile before factories or payload I/O run."""
        authority_identity = _reference_qualification_identity(
            BackendRole.AUTHORITY.value, self.authority
        )
        payload_identity = _reference_qualification_identity(
            BackendRole.PAYLOAD.value, self.payload
        )
        try:
            return BUILTIN_QUALIFIED_TOPOLOGY_PROFILES[
                (authority_identity, payload_identity)
            ]
        except KeyError as error:
            raise CompositionValidationError(
                "Unsupported topology pairing: "
                f"authority={authority_identity!r}, payload={payload_identity!r}"
            ) from error


def resolve_metadata_role(implementation: str) -> MetadataRole:
    """Return the current explicit metadata role for a named family."""
    roles = {
        "memory": MetadataRole(BackendRole.AUTHORITY.value),
        "sqlite": MetadataRole(BackendRole.AUTHORITY.value),
        "json": MetadataRole(BackendRole.PROJECTION.value),
        "postgresql": MetadataRole(BackendRole.AUTHORITY.value),
    }
    try:
        return roles[implementation]
    except KeyError as error:
        raise CompositionValidationError(
            f"Metadata implementation {implementation!r} has no Phase 4 role"
        ) from error


def _normalize_role(role: str | BackendRole) -> str:
    value = role.value if isinstance(role, BackendRole) else role
    if value not in {item.value for item in BackendRole}:
        raise CompositionValidationError(f"Unsupported participant role: {value!r}")
    return value


def _as_ref(value: BackendRef | object) -> BackendRef:
    return value if isinstance(value, BackendRef) else BackendRef(instance=value)


def _reference_qualification_identity(role: str, reference: BackendRef) -> str:
    """Read one explicit support identity without constructing a participant."""
    if reference.name is not None:
        return reference.name
    assert reference.instance is not None
    identity = getattr(reference.instance, "qualification_identity", None)
    if not isinstance(identity, str) or not identity:
        raise CompositionValidationError(
            f"Injected {role} participant must declare a non-empty "
            "qualification_identity"
        )
    return identity


def _projection_reference_identity(reference: BackendRef) -> tuple[str, object]:
    """Return a local duplicate detector without constructing projections."""
    if reference.name is not None:
        return ("name", reference.name)
    assert reference.instance is not None
    return ("instance", id(reference.instance))


def _resolve_ref(
    role: str,
    reference: BackendRef,
    registry: RoleRegistry,
    owned: list[object],
) -> object:
    if reference.instance is not None:
        participant = reference.instance
        if reference.ownership is Ownership.STORE:
            _record_owned_participant(owned, participant)
        _validate_participant_role(role, participant)
        return participant

    assert reference.name is not None
    participant = registry.construct(role, reference.name, reference.options)
    if reference.ownership is Ownership.STORE:
        _record_owned_participant(owned, participant)
    _validate_participant_role(role, participant)
    return participant


def _validate_participant_role(role: str, participant: object) -> None:
    """Require one narrow role protocol before any store I/O can begin."""
    if participant is None:
        raise CompositionValidationError(f"{role} participant cannot be None")

    is_authority = isinstance(participant, LifecycleAuthority)
    is_payload = isinstance(participant, PayloadGenerationIOProvider)
    is_projection = isinstance(participant, ProjectionSink)

    if role == BackendRole.PAYLOAD.value:
        if is_authority or is_projection:
            raise CompositionValidationError(
                "A payload participant cannot fill another role"
            )
        if not is_payload:
            raise CompositionValidationError(
                "A payload participant must provide guarded generation I/O"
            )
        return

    if role == BackendRole.AUTHORITY.value:
        if is_payload or is_projection:
            raise CompositionValidationError(
                "An authority participant cannot fill another role"
            )
        if not is_authority:
            raise CompositionValidationError(
                "An authority participant must satisfy LifecycleAuthority"
            )
        return

    if role == BackendRole.PROJECTION.value:
        if is_authority or is_payload:
            raise CompositionValidationError(
                "A projection participant cannot fill another role"
            )
        if not is_projection:
            raise CompositionValidationError(
                "A projection participant must satisfy ProjectionSink"
            )
        return

    raise CompositionValidationError(f"Unsupported participant role: {role!r}")


def _record_and_validate_injected_participants(
    topology: StoreTopology, owned: list[object]
) -> None:
    """Capture every transferred injection before its role validation can fail."""
    references = (
        (BackendRole.PAYLOAD.value, topology.payload),
        (BackendRole.AUTHORITY.value, topology.authority),
        *(
            (BackendRole.PROJECTION.value, projection)
            for projection in topology.projections
        ),
    )
    for _role, reference in references:
        if reference.instance is not None and reference.ownership is Ownership.STORE:
            _record_owned_participant(owned, reference.instance)
    for role, reference in references:
        if reference.instance is not None:
            _validate_participant_role(role, reference.instance)


def _record_owned_participant(owned: list[object], participant: object) -> None:
    """Record an owned object once by identity in first-acquisition order."""
    if not any(existing is participant for existing in owned):
        owned.append(participant)


def _close_owned_participants(
    participants: tuple[object, ...] | list[object], *, suppress_errors: bool = False
) -> None:
    """Close one local ledger in reverse acquisition order without sharing state."""
    first_error: Exception | None = None
    for participant in reversed(participants):
        close = getattr(participant, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None and not suppress_errors:
        raise first_error


def _capability_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise CompositionValidationError(f"{field_name} capability must be a bool")
    return value


def _capability_scope(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value not in _SCOPE_RANK:
        raise CompositionValidationError(
            f"{field_name} capability must be one of {sorted(_SCOPE_RANK)}"
        )
    return value


def _weakest_scope(*scopes: str) -> str:
    return min(scopes, key=_SCOPE_RANK.__getitem__)


def _compose_capabilities(
    payload: object,
    authority: object,
    projections: tuple[object, ...],
) -> TopologyCapabilities:
    """Compose guarantees only from the participant that can supply each one."""
    payload_capabilities = ParticipantCapabilities.from_participant(
        payload, BackendRole.PAYLOAD.value
    )
    authority_capabilities = ParticipantCapabilities.from_participant(
        authority, BackendRole.AUTHORITY.value
    )
    projection_capabilities = tuple(
        ParticipantCapabilities.from_participant(
            projection, BackendRole.PROJECTION.value
        )
        for projection in projections
    )
    return _compose_capability_values(
        payload_capabilities,
        authority_capabilities,
        projection_capabilities,
    )


def _compose_capability_values(
    payload_capabilities: ParticipantCapabilities,
    authority_capabilities: ParticipantCapabilities,
    projection_capabilities: tuple[ParticipantCapabilities, ...],
) -> TopologyCapabilities:
    """Compose already-declared participant capabilities conservatively."""
    projection_supported = bool(projection_capabilities)
    return TopologyCapabilities(
        durable=payload_capabilities.durable and authority_capabilities.durable,
        process_scope=_weakest_scope(
            payload_capabilities.process_scope, authority_capabilities.process_scope
        ),
        host_scope=_weakest_scope(
            payload_capabilities.host_scope, authority_capabilities.host_scope
        ),
        transaction_scope=authority_capabilities.transaction_scope,
        exact_cas=authority_capabilities.exact_cas,
        immutable_generations=payload_capabilities.immutable_generations,
        streaming=payload_capabilities.streaming,
        listing=payload_capabilities.listing,
        portable_query=authority_capabilities.portable_query,
        canonical_scan=authority_capabilities.canonical_scan,
        index_acceleration=authority_capabilities.index_acceleration,
        projection_refresh=projection_supported
        and all(item.projection_refresh for item in projection_capabilities),
        projection_rebuild=projection_supported
        and all(item.projection_rebuild for item in projection_capabilities),
        online_rebuild=projection_supported
        and all(item.online_rebuild for item in projection_capabilities),
        offline_rebuild=projection_supported
        and all(item.offline_rebuild for item in projection_capabilities),
    )


def _capabilities_from_references(
    topology: StoreTopology, registry: RoleRegistry
) -> TopologyCapabilities:
    """Validate minima from declarations before a named factory can perform I/O."""

    def capabilities_for(role: str, reference: BackendRef) -> ParticipantCapabilities:
        if reference.instance is not None:
            return ParticipantCapabilities.from_participant(reference.instance, role)
        assert reference.name is not None
        return registry.capabilities(role, reference.name)

    payload = capabilities_for(BackendRole.PAYLOAD.value, topology.payload)
    authority = capabilities_for(BackendRole.AUTHORITY.value, topology.authority)
    projections = tuple(
        capabilities_for(BackendRole.PROJECTION.value, projection)
        for projection in topology.projections
    )
    return _compose_capability_values(payload, authority, projections)


def allowed_progress_outcomes(authority_kind: str) -> set[str]:
    """Report the topology-qualified progress outcomes callers must handle."""
    if authority_kind == "sqlite-local":
        return {"success", "conflict", "retryable_timeout"}
    if authority_kind == "memory":
        return {"success", "conflict"}
    if authority_kind == "postgresql-remote":
        return {
            "success",
            "conflict",
            "retryable_serialization",
            "retryable_deadlock",
            "retryable_lock_timeout",
            "retryable_statement_timeout",
            "retryable_connection_timeout",
        }
    raise CompositionValidationError(
        f"No declared progress behavior for authority kind {authority_kind!r}"
    )


__all__ = [
    "BackendRef",
    "BackendRole",
    "CapabilityMinimum",
    "CapabilityRequirementError",
    "CompositionValidationError",
    "BUILTIN_QUALIFIED_TOPOLOGY_PROFILES",
    "MetadataRole",
    "Ownership",
    "PayloadGenerationIOProvider",
    "PayloadTransportObservationProvider",
    "ParticipantCapabilities",
    "ProjectionRole",
    "ProjectionSink",
    "QualifiedTopologyProfile",
    "ResolvedTopology",
    "RoleRegistration",
    "RoleRegistry",
    "StoreTopology",
    "TopologyCapabilities",
    "TopologyQualificationRequirements",
    "allowed_progress_outcomes",
    "qualified_topology_profiles",
    "resolve_metadata_role",
]
