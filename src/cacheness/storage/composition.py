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

from .backends.blob_backends import FilesystemBlobBackend, InMemoryBlobBackend
from .lifecycle_authority import LifecycleAuthority
from .memory_lifecycle_authority import InMemoryLifecycleAuthority
from .sqlite_lifecycle_authority import SqliteLifecycleAuthority


class CompositionValidationError(ValueError):
    """Raised when a topology cannot be resolved without ambiguity."""


class BackendRole(str, Enum):
    """The only participant roles accepted by a store topology."""

    PAYLOAD = "payload"
    AUTHORITY = "authority"
    PROJECTION = "projection"


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
        if not isinstance(self.options, Mapping):
            raise CompositionValidationError("BackendRef options must be a mapping")
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class RoleRegistration:
    """A single named factory declaration for a participant role."""

    role: str
    name: str
    factory: Callable[..., object]

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
        self.register(BackendRole.PAYLOAD.value, "filesystem", FilesystemBlobBackend)
        self.register(
            BackendRole.AUTHORITY.value,
            "memory",
            InMemoryLifecycleAuthority,
        )
        self.register(
            BackendRole.AUTHORITY.value,
            "sqlite",
            _construct_sqlite_authority,
        )
        self.register(
            BackendRole.PROJECTION.value,
            "json",
            _construct_json_projection,
        )
        self.register(
            BackendRole.PROJECTION.value,
            "postgresql",
            _construct_postgresql_projection,
        )

    def register(
        self,
        role: str | BackendRole,
        name: str,
        factory: Callable[..., object],
        *,
        replace: bool = False,
    ) -> None:
        """Register a factory under exactly one participant role."""
        normalized_role = _normalize_role(role)
        if not isinstance(name, str) or not name:
            raise CompositionValidationError("Participant registration name must be a string")
        if not callable(factory):
            raise CompositionValidationError("Participant registration factory must be callable")
        key = (normalized_role, name)
        if key in self._entries and not replace:
            raise CompositionValidationError(
                f"A {normalized_role} participant named {name!r} is already registered"
            )
        self._entries[key] = RoleRegistration(normalized_role, name, factory)

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


@dataclass
class ResolvedTopology:
    """Resolved participants plus the ownership ledger used by later lifecycle work."""

    payload: object
    authority: object
    projections: tuple[object, ...]
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
    from cacheness.metadata import JsonBackend

    return JsonBackend(Path(metadata_file), **options)


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
    minimum_capabilities: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _as_ref(self.payload))
        object.__setattr__(self, "authority", _as_ref(self.authority))
        object.__setattr__(
            self,
            "projections",
            tuple(_as_ref(projection) for projection in self.projections),
        )
        if not isinstance(self.minimum_capabilities, Mapping):
            raise CompositionValidationError("minimum_capabilities must be a mapping")
        object.__setattr__(self, "minimum_capabilities", dict(self.minimum_capabilities))

    def resolve(self, registry: RoleRegistry | None = None) -> ResolvedTopology:
        """Resolve each role once and unwind only owned participants on failure."""
        active_registry = RoleRegistry() if registry is None else registry
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
            return ResolvedTopology(payload, authority, projections, tuple(owned))
        except BaseException:
            for participant in reversed(owned):
                close = getattr(participant, "close", None)
                if callable(close):
                    close()
            raise


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
        if reference.transfer_ownership:
            owned.append(participant)
        return participant

    assert reference.name is not None
    participant = registry.construct(role, reference.name, reference.options)
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
    if role != BackendRole.PROJECTION.value and participant is None:
        raise CompositionValidationError(f"{role} participant cannot be None")


__all__ = [
    "BackendRef",
    "BackendRole",
    "CompositionValidationError",
    "MetadataRole",
    "ResolvedTopology",
    "RoleRegistration",
    "RoleRegistry",
    "StoreTopology",
    "resolve_metadata_role",
]
