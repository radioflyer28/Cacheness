"""Current topology qualification contracts for direct BlobStore composition."""

from __future__ import annotations

import pytest

from cacheness.storage.composition import BackendRef, StoreTopology


def test_memory_authority_and_payload_form_the_supported_local_profile() -> None:
    """The in-process topology is an explicit qualified composition."""

    topology = StoreTopology(
        payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
    )

    profile = topology.qualification_report()

    assert profile.pair == ("memory", "memory")
    assert profile.requirements.coordination_scope == "one_process"


def test_unqualified_authority_payload_pair_is_rejected_before_io() -> None:
    """A convenient backend name cannot imply unsupported topology guarantees."""

    topology = StoreTopology(
        payload=BackendRef(name="s3"), authority=BackendRef(name="sqlite")
    )

    with pytest.raises(ValueError, match="Unsupported topology pairing"):
        topology.qualification_report()
