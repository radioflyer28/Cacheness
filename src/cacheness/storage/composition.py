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

from .backends.blob_backends import FilesystemBlobBackend, InMemoryBlobBackend
from .lifecycle_authority import LifecycleAuthority
from .memory_lifecycle_authority import InMemoryLifecycleAuthority
from .sqlite_lifecycle_authority import SqliteLifecycleAuthority


class CompositionValidationError(ValueError):
    """Raised when a topology cannot be resolved without ambiguity."""


class CapabilityRequirementError(CompositionValidationError):
    """Raised before lifecycle I/O when a topology misses a required guarantee."""


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
    def from_participant(cls, participant: object, role: str) -> "ParticipantCapabilities":
        """Normalize a participant's local declaration without inferring strength."""
        declared = participant if isinstance(participant, Mapping) else getattr(
            participant, "topology_capabilities", None
        )
        if isinstance(declared, cls):
            return declared
        raw = declared if declared is not None else getattr(participant, "capabilities", {})
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
            process_scope=_capability_scope(get("process_scope", "process"), "process_scope"),
            host_scope=_capability_scope(get("host_scope", "process"), "host_scope"),
            transaction_scope=get(
                "transaction_scope",
                "authority" if role == BackendRole.AUTHORITY.value and transactional else "none",
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
        unsupported = set(self.requirements) - set(TopologyCapabilities.__dataclass_fields__)
        if unsupported:
            raise CompositionValidationError(
                f"Unsupported capability minimum: {sorted(unsupported)}"
            )
        object.__setattr__(self, "requirements", MappingProxyType(dict(self.requirements)))

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
        return self.kind == BackendRole.AUTHORITY.value and operation in authority_operations


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
            raise CompositionValidationError("BackendRef name must be a non-empty string")
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
                raise CompositionValidationError("ownership must be caller or store") from error
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
        # Built-ins take the same path as application registrations.  JSON and
        # PostgreSQL are deliberately metadata projections, never Phase 4
        # lifecycle authorities.
        self.register(BackendRole.PAYLOAD.value, "memory", InMemoryBlobBackend)
        self.register(
            BackendRole.PAYLOAD.value,
            "filesystem",
            FilesystemBlobBackend,
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
                "portable_query": False,
                "canonical_scan": True,
                "index_acceleration": False,
            },
        )
        self.register(
            BackendRole.PROJECTION.value,
            "json",
            _construct_json_projection,
            capabilities={},
        )
        self.register(
            BackendRole.PROJECTION.value,
            "postgresql",
            _construct_postgresql_projection,
            capabilities={},
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
            raise CompositionValidationError("Participant registration name must be a string")
        if not callable(factory):
            raise CompositionValidationError("Participant registration factory must be callable")
        if capabilities is None:
            capabilities = getattr(factory, "topology_capabilities", None)
        if capabilities is not None and not isinstance(capabilities, ParticipantCapabilities):
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
        """Construct a named participant through the same role registry path."""
        participant = self.resolve(role, name).construct(options)
        _validate_participant_role(_normalize_role(role), participant)
        return participant

    def capabilities(self, role: str | BackendRole, name: str) -> ParticipantCapabilities:
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
    _owned_in_creation_order: tuple[object, ...] = ()
    _closed: bool = False

    def close(self) -> None:
        """Close only owned participants, exactly once, in reverse creation order."""
        if self._closed:
            return
        self._closed = True
        for participant in reversed(self._owned_in_creation_order):
            close = getattr(participant, "close", None)
            if callable(close):
                close()


def _construct_sqlite_authority(*, root: str | Path, **options: object) -> object:
    """Build a local authority only when a topology supplies its explicit root."""
    return SqliteLifecycleAuthority.for_root(Path(root), **options)


def _construct_json_projection(*, metadata_file: str | Path, **options: object) -> object:
    """Build the JSON carrier only as a derived projection participant."""
    from cacheness.metadata import JsonProjection

    return JsonProjection(Path(metadata_file), **options)


def _construct_postgresql_projection(**options: object) -> object:
    """Build PostgreSQL only as a deferred derived-projection participant."""
    from .backends.postgresql_backend import PostgresBackend

    return PostgresBackend(**options)


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
        object.__setattr__(
            self,
            "projections",
            tuple(_as_ref(projection) for projection in self.projections),
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
        if self.minimum_capabilities.requirements:
            try:
                self.minimum_capabilities.require(
                    _capabilities_from_references(self, active_registry)
                )
            except BaseException:
                self._close_transferred_injections()
                raise
        owned: list[object] = []
        try:
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
                tuple(owned),
            )
        except BaseException:
            for participant in reversed(owned):
                close = getattr(participant, "close", None)
                if callable(close):
                    close()
            raise

    def _close_transferred_injections(self) -> None:
        """Release transfer-owned injections if preflight aborts construction."""
        references = (self.payload, self.authority, *self.projections)
        for reference in reversed(references):
            if (
                reference.instance is not None
                and reference.ownership is Ownership.STORE
            ):
                close = getattr(reference.instance, "close", None)
                if callable(close):
                    close()

    def capability_report(self) -> TopologyCapabilities:
        """Return the resolved topology's capability report without retaining it."""
        resolved = self.resolve()
        try:
            return resolved.capabilities
        finally:
            resolved.close()


def resolve_metadata_role(implementation: str) -> MetadataRole:
    """Return the explicit Phase 4 metadata role for a named family."""
    roles = {
        "memory": MetadataRole(BackendRole.AUTHORITY.value),
        "sqlite": MetadataRole(BackendRole.AUTHORITY.value),
        "json": MetadataRole(BackendRole.PROJECTION.value),
        "postgresql": MetadataRole(BackendRole.PROJECTION.value),
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


def _resolve_ref(
    role: str,
    reference: BackendRef,
    registry: RoleRegistry,
    owned: list[object],
) -> object:
    if reference.instance is not None:
        participant = reference.instance
        _validate_participant_role(role, participant)
        if reference.ownership is Ownership.STORE:
            owned.append(participant)
        return participant

    assert reference.name is not None
    participant = registry.construct(role, reference.name, reference.options)
    if reference.ownership is Ownership.STORE:
        owned.append(participant)
    return participant


def _validate_participant_role(role: str, participant: object) -> None:
    """Reject known cross-role participants before any store I/O occurs."""
    is_authority = isinstance(participant, LifecycleAuthority)
    is_payload = isinstance(participant, (FilesystemBlobBackend, InMemoryBlobBackend))
    if role == BackendRole.AUTHORITY.value and is_payload:
        raise CompositionValidationError("A payload participant cannot be an authority")
    if role == BackendRole.PAYLOAD.value and is_authority:
        raise CompositionValidationError("A lifecycle authority cannot be a payload")
    if role == BackendRole.PAYLOAD.value and not isinstance(
        participant, PayloadGenerationIOProvider
    ):
        raise CompositionValidationError(
            "A payload participant must provide guarded generation I/O"
        )
    if role != BackendRole.PROJECTION.value and participant is None:
        raise CompositionValidationError(f"{role} participant cannot be None")


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
        ParticipantCapabilities.from_participant(projection, BackendRole.PROJECTION.value)
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
    raise CompositionValidationError(
        f"No declared progress behavior for authority kind {authority_kind!r}"
    )


__all__ = [
    "BackendRef",
    "BackendRole",
    "CapabilityMinimum",
    "CapabilityRequirementError",
    "CompositionValidationError",
    "MetadataRole",
    "Ownership",
    "PayloadGenerationIOProvider",
    "ParticipantCapabilities",
    "ProjectionRole",
    "ProjectionSink",
    "ResolvedTopology",
    "RoleRegistration",
    "RoleRegistry",
    "StoreTopology",
    "TopologyCapabilities",
    "allowed_progress_outcomes",
    "resolve_metadata_role",
]
