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
    allowed_progress_outcomes,
    resolve_metadata_role,
)
from cacheness.storage.backends.blob_backends import InMemoryBlobBackend
from cacheness.storage.lifecycle import AuthorityLifecycleEngine
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
from cacheness.error_handling import CacheBlobBackendError


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


def test_builtin_remote_roles_expose_only_the_qualified_participants() -> None:
    """Remote construction names identify the completed narrow role adapters."""
    registry = RoleRegistry()

    payload = registry.resolve("payload", "s3")
    authority = registry.resolve("authority", "postgresql")

    assert payload.capabilities is not None
    assert payload.capabilities.process_scope == "multi_host"
    assert payload.capabilities.streaming is True
    assert authority.capabilities is not None
    assert authority.capabilities.transaction_scope == "authority"
    assert authority.capabilities.exact_cas is True
    assert allowed_progress_outcomes("postgresql-remote") == {
        "success",
        "conflict",
        "retryable_serialization",
        "retryable_deadlock",
        "retryable_lock_timeout",
        "retryable_statement_timeout",
        "retryable_connection_timeout",
    }


class _StaticRemoteManifestKey:
    """Application-owned shared key material used by independent remote clients."""

    def __init__(self, key: bytes) -> None:
        self._key = key

    def get_key(self) -> bytes:
        return self._key

    def get_or_initialize_new_store(self) -> bytes:
        return self._key

    def initialize_new_store(self) -> bytes:
        return self._key


class _RemotePayload(InMemoryBlobBackend):
    """Deterministic participant boundary used without claiming live S3 evidence."""

    qualification_identity = "s3"
    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "immutable_generations": True,
        "streaming": True,
        "listing": True,
    }


class _RemoteAuthority(InMemoryLifecycleAuthority):
    """Deterministic authority boundary used without a PostgreSQL service."""

    qualification_identity = "postgresql"
    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "transaction_scope": "authority",
        "exact_cas": True,
        "portable_query": True,
        "canonical_scan": True,
        "index_acceleration": True,
    }


def test_remote_profile_requires_external_key_and_retains_one_engine(tmp_path) -> None:
    """A remote profile never falls back to a per-host signing-key file."""
    registry = RoleRegistry()
    registry.register(
        "payload",
        "s3",
        _RemotePayload,
        capabilities=_RemotePayload.topology_capabilities,
    )
    registry.register(
        "authority",
        "postgresql",
        _RemoteAuthority,
        capabilities=_RemoteAuthority.topology_capabilities,
    )
    topology = StoreTopology(
        payload=BackendRef(name="s3"),
        authority=BackendRef(name="postgresql"),
        role_registry=registry,
    )

    with pytest.raises(CacheBlobBackendError, match="external manifest signing key"):
        BlobStore(topology, cache_dir=tmp_path / "remote-default-key")

    with BlobStore(
        topology,
        cache_dir=tmp_path / "remote-explicit-key",
        manifest_key_provider=_StaticRemoteManifestKey(b"r" * 32),
    ) as store:
        assert type(store.lifecycle) is AuthorityLifecycleEngine
        assert store.payload_backend.qualification_identity == "s3"
        assert store.lifecycle_authority.qualification_identity == "postgresql"
        assert store.topology.qualified_profile.pair == ("postgresql", "s3")


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
        StoreTopology(payload=BackendRef(name="memory"), authority=None)
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
