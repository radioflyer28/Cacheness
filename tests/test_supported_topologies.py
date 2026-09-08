"""Public contract tests for the immutable supported-topology catalog."""

from __future__ import annotations

from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, RoleRegistry, StoreTopology
from cacheness.storage.lifecycle import AuthorityLifecycleEngine


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
        assert store.qualified_profile is profile
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
