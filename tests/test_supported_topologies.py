"""Public contract tests for the immutable supported-topology catalog."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import (
    BackendRef,
    CompositionValidationError,
    RoleRegistry,
    StoreTopology,
    resolve_metadata_role,
)
from cacheness.storage.lifecycle import AuthorityLifecycleEngine
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority


def test_memory_profile_resolves_before_the_public_store_round_trip(tmp_path) -> None:
    """The supported memory pair uses the existing lifecycle engine unchanged."""
    topology = StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )

    profile = topology.qualification_report()

    assert profile.authority_identity == "memory"
    assert profile.payload_identity == "memory"
    assert profile.requirements.coordination_scope == "one_process"
    assert profile.requirements.evidence_requirement_id == "local-memory-contract"

    with BlobStore(topology, cache_dir=tmp_path / "memory-store") as store:
        assert isinstance(store.lifecycle, AuthorityLifecycleEngine)
        assert store.topology.qualified_profile is profile
        assert store.put({"answer": 42}, key="answer") == "answer"
        assert store.get("answer") == {"answer": 42}


def test_builtin_catalog_has_only_the_three_declared_reference_pairs() -> None:
    """Registration is constructibility; the catalog is the support boundary."""
    from cacheness.storage.composition import BUILTIN_QUALIFIED_TOPOLOGY_PROFILES

    assert set(BUILTIN_QUALIFIED_TOPOLOGY_PROFILES) == {
        ("memory", "memory"),
        ("sqlite", "filesystem"),
        ("postgresql", "s3"),
    }
    assert len(BUILTIN_QUALIFIED_TOPOLOGY_PROFILES) == 3

    registry = RoleRegistry()
    assert registry.resolve("authority", "sqlite").capabilities is not None
    assert registry.resolve("authority", "sqlite").capabilities.portable_query is True


class _QualifiedPayload:
    """A structurally valid payload participant for pre-I/O rejection tests."""

    def __init__(self, identity: str | None) -> None:
        if identity is not None:
            self.qualification_identity = identity

    def materialize_handler_io(self) -> object:
        return object()


class _QualifiedAuthority(InMemoryLifecycleAuthority):
    """A structurally valid authority participant with optional identity."""

    def __init__(self, identity: str | None) -> None:
        super().__init__()
        if identity is not None:
            self.qualification_identity = identity


class _Projection:
    """Derived-only projection used to prove it does not alter profile identity."""

    projection_name = "test-projection"

    def apply_projection_batch(self, batch: object) -> None:
        del batch

    def save_projection_checkpoint(self, checkpoint: object) -> None:
        del checkpoint

    def load_projection_checkpoint(self) -> None:
        return None


def _registry_with_factory_spies() -> tuple[RoleRegistry, list[tuple[str, str]]]:
    """Register structurally valid factories whose calls expose early rejection."""
    calls: list[tuple[str, str]] = []
    registry = RoleRegistry()

    def payload_factory(name: str) -> Callable[[], _QualifiedPayload]:
        def factory() -> _QualifiedPayload:
            calls.append(("payload", name))
            return _QualifiedPayload(name)

        return factory

    def authority_factory(name: str) -> Callable[[], _QualifiedAuthority]:
        def factory() -> _QualifiedAuthority:
            calls.append(("authority", name))
            return _QualifiedAuthority(name)

        return factory

    for payload_name in ("memory", "filesystem", "s3"):
        registry.register(
            "payload",
            payload_name,
            payload_factory(payload_name),
            replace=True,
        )
    for authority_name in ("memory", "sqlite", "postgresql"):
        registry.register(
            "authority",
            authority_name,
            authority_factory(authority_name),
            replace=authority_name != "postgresql",
        )
    return registry, calls


@pytest.mark.parametrize(
    ("authority_name", "payload_name"),
    [
        ("memory", "filesystem"),
        ("memory", "s3"),
        ("sqlite", "memory"),
        ("sqlite", "s3"),
        ("postgresql", "memory"),
        ("postgresql", "filesystem"),
    ],
)
def test_unqualified_cartesian_pairs_reject_before_factories(
    authority_name: str, payload_name: str
) -> None:
    """A registered and structurally valid pair is still not a support claim."""
    registry, calls = _registry_with_factory_spies()

    with pytest.raises(CompositionValidationError, match="Unsupported topology pairing"):
        StoreTopology(
            payload=BackendRef(name=payload_name),
            authority=BackendRef(name=authority_name),
            role_registry=registry,
        ).resolve()

    assert calls == []


def test_incomplete_or_duplicate_topology_declarations_fail_without_factories() -> None:
    """Null, partial, and duplicate declarations remain descriptor errors."""
    registry, calls = _registry_with_factory_spies()

    with pytest.raises(CompositionValidationError):
        StoreTopology(payload=None, authority=BackendRef(name="memory"))
    with pytest.raises(CompositionValidationError):
        BackendRef(name="")
    with pytest.raises(CompositionValidationError, match="qualification_identity"):
        StoreTopology(
            payload=_QualifiedPayload(None),
            authority=BackendRef(name="memory"),
            role_registry=registry,
        ).resolve()
    with pytest.raises(CompositionValidationError, match="Duplicate projection"):
        StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
            projections=(BackendRef(name="json"), BackendRef(name="json")),
            role_registry=registry,
        )

    assert calls == []


def test_registration_and_projection_order_do_not_change_support_identity() -> None:
    """JSON is derived-only and profile lookup ignores registration order."""
    first_registry, _ = _registry_with_factory_spies()
    second_registry = RoleRegistry()
    second_registry.register("authority", "postgresql", lambda: _QualifiedAuthority("postgresql"))
    second_registry.register("payload", "s3", lambda: _QualifiedPayload("s3"))

    base = StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
        role_registry=first_registry,
    ).qualification_report()
    with_json = StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
        projections=(_Projection(),),
        role_registry=second_registry,
    ).qualification_report()

    assert with_json is base
    assert resolve_metadata_role("json").authorizes("promote_catalog") is False
