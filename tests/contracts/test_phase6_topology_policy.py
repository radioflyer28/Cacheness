"""Topology and ownership contracts for the UnifiedCache policy facade.

The remote profile here is a deterministic candidate built from in-memory
participants with the published remote capability identities.  It exercises
the same cache-to-BlobStore call graph without claiming PostgreSQL or Amazon
S3 live-service qualification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheBlobStoreClosedError
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.backends.blob_backends import InMemoryBlobBackend
from cacheness.storage.composition import (
    BackendRef,
    RoleRegistry,
    StoreTopology,
    qualified_topology_profiles,
)
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority


class _SharedRemoteManifestKey:
    """Provide deterministic application-owned key material for a candidate."""

    def get_key(self) -> bytes:
        return b"r" * 32

    def get_or_initialize_new_store(self) -> bytes:
        return self.get_key()

    def initialize_new_store(self) -> bytes:
        return self.get_key()


class _CandidateS3Payload(InMemoryBlobBackend):
    """S3-shaped candidate payload with no connection to a live service."""

    qualification_identity = "s3"
    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "immutable_generations": True,
        "streaming": True,
        "listing": True,
    }


class _CandidatePostgresqlAuthority(InMemoryLifecycleAuthority):
    """PostgreSQL-shaped authority used only for deterministic policy tests."""

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


def _local_topology(profile: str, root: Path) -> StoreTopology:
    """Create one Phase-5-qualified local topology."""

    if profile == "memory-memory":
        return StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
        )
    if profile == "sqlite-filesystem":
        return StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        )
    raise ValueError(f"Unknown local profile: {profile}")


@pytest.mark.parametrize("profile", ("memory-memory", "sqlite-filesystem"))
def test_local_profiles_share_one_explicit_cache_lifecycle(
    tmp_path: Path, profile: str
) -> None:
    """Local profiles use the same initialized policy/store lifecycle."""

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path / profile)),
        store=_local_topology(profile, tmp_path / profile / "payloads"),
    )
    try:
        cache.initialize()
        stored = cache.put({"profile": profile}, request_id=profile)

        assert cache.store is cache._cache_blob_store
        assert cache.store.lifecycle is cache.store._authority_lifecycle
        assert cache.lookup(cache_key=stored.receipt.key).value == {"profile": profile}
        assert cache.invalidate(cache_key=stored.receipt.key).removed == 1
    finally:
        cache.close()


def test_deterministic_remote_candidate_uses_the_same_policy_call_graph(
    tmp_path: Path,
) -> None:
    """The remote profile is an explicit candidate, never live evidence."""

    topology = StoreTopology(
        payload=_CandidateS3Payload(),
        authority=_CandidatePostgresqlAuthority(),
    )
    store = BlobStore(
        topology,
        cache_dir=tmp_path / "candidate-store",
        manifest_key_provider=_SharedRemoteManifestKey(),
    )
    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path / "cache")),
        store=store,
    )
    try:
        store.initialize()
        stored = cache.put({"profile": "candidate"}, request_id="candidate")

        assert cache.store is store
        assert cache.store.lifecycle is store.lifecycle
        assert cache.lookup(cache_key=stored.receipt.key).value == {
            "profile": "candidate"
        }
        profile = store.topology.qualified_profile
        assert profile.pair == ("postgresql", "s3")
        assert profile.requirements.evidence_requirement_id == "live-postgresql-amazon-s3"
        assert profile.requirements.evidence_schema_id == "phase5-live-service-evidence-v1"
    finally:
        cache.close()
        store.close()


def test_profiles_are_declared_in_stable_authority_payload_order() -> None:
    """Profile discovery is declaration order, not registration/import order."""

    assert tuple(qualified_topology_profiles()) == (
        ("memory", "memory"),
        ("sqlite", "filesystem"),
        ("postgresql", "s3"),
    )


def test_unsupported_topology_rejects_before_participant_construction(
    tmp_path: Path,
) -> None:
    """An invalid adjacency cannot reach cache or BlobStore participant I/O."""

    constructed: list[str] = []
    registry = RoleRegistry()
    registry.register(
        "payload",
        "unqualified-payload",
        lambda: constructed.append("payload"),
        capabilities={"immutable_generations": True, "streaming": True, "listing": True},
    )
    registry.register(
        "authority",
        "unqualified-authority",
        lambda: constructed.append("authority"),
        capabilities={
            "durable": True,
            "process_scope": "process",
            "host_scope": "process",
            "transaction_scope": "authority",
            "exact_cas": True,
            "portable_query": True,
            "canonical_scan": True,
            "index_acceleration": False,
        },
    )
    topology = StoreTopology(
        payload=BackendRef(name="unqualified-payload"),
        authority=BackendRef(name="unqualified-authority"),
        role_registry=registry,
    )

    with pytest.raises(ValueError, match="Unsupported topology pairing"):
        UnifiedCache(
            CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path / "cache")),
            store=topology,
        )

    assert constructed == []


def test_injected_store_remains_caller_initialized_and_caller_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Facade initialization and close never take ownership of an injected store."""

    store = BlobStore(
        _local_topology("memory-memory", tmp_path / "payloads"),
        cache_dir=tmp_path / "store",
    )
    store.initialize()
    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path / "cache")),
        store=store,
    )
    initialize_calls = 0
    close_calls = 0

    def unexpected_initialize() -> None:
        nonlocal initialize_calls
        initialize_calls += 1
        raise AssertionError("cache must not initialize a caller-owned store")

    original_close = store.close

    def observed_close() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    monkeypatch.setattr(store, "initialize", unexpected_initialize)
    monkeypatch.setattr(store, "close", observed_close)
    try:
        cache.initialize()
        cache.close()

        assert initialize_calls == 0
        assert close_calls == 0
        assert store.get("missing") is None
    finally:
        store.close()


def test_cache_ownership_uses_constructor_form_not_a_boolean_flag() -> None:
    """Ownership is represented by the selected store form, never a flag."""

    source = (
        Path(__file__).resolve().parents[2] / "src" / "cacheness" / "core.py"
    ).read_text(encoding="utf-8")

    assert "_owns_store" not in source


@pytest.mark.parametrize("store_form", ("injected", "topology"))
def test_cache_close_closes_the_facade_boundary_without_releasing_injected_store(
    tmp_path: Path, store_form: str
) -> None:
    """Close blocks policy observers while preserving caller-owned storage."""

    topology = _local_topology("memory-memory", tmp_path / "payloads")
    if store_form == "injected":
        store = BlobStore(topology, cache_dir=tmp_path / "caller-store")
        store.initialize()
        cache = UnifiedCache(
            CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path / "cache")),
            store=store,
        )
    else:
        store = None
        cache = UnifiedCache(
            CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path / "cache")),
            store=topology,
        )
        cache.initialize()

    cache.close()

    with pytest.raises(CacheBlobStoreClosedError, match="Cache is closed"):
        cache.statistics()

    if store is not None:
        try:
            assert store.get("missing") is None
        finally:
            store.close()
