"""Red contracts for truthful, participant-derived topology capabilities."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import pytest


COMPOSITION_MODULE = "cacheness.storage.composition"


def _composition():
    assert find_spec(COMPOSITION_MODULE) is not None, (
        "Phase 4 must calculate capabilities from active topology participants"
    )
    return import_module(COMPOSITION_MODULE)


class _Payload:
    def __init__(self) -> None:
        self.stage_calls = 0

    capabilities = {
        "durable": False,
        "process_scope": "process",
        "host_scope": "process",
        "immutable_generations": True,
        "streaming": False,
        "listing": False,
    }

    def stage(self) -> None:
        self.stage_calls += 1


class _Authority:
    capabilities = {
        "durable": False,
        "process_scope": "process",
        "host_scope": "process",
        "compare_and_swap": True,
        "transactional": True,
        "portable_query": True,
    }


class _Projection:
    capabilities = {"derived_only": True, "online_rebuild": False, "offline_rebuild": True}


def test_memory_topology_is_explicitly_ephemeral_and_same_process_only() -> None:
    composition = _composition()

    report = composition.StoreTopology(payload=_Payload(), authority=_Authority()).capability_report()

    assert report.durable is False
    assert report.process_scope == "process"
    assert report.host_scope == "process"


def test_composed_report_intersects_actual_participant_capabilities_not_names() -> None:
    composition = _composition()

    report = composition.StoreTopology(
        payload=_Payload(), authority=_Authority(), projections=(_Projection(),)
    ).capability_report()

    assert report.immutable_generations is True
    assert report.exact_cas is True
    assert report.streaming is False
    assert report.online_rebuild is False


@pytest.mark.parametrize(
    "minimum",
    [
        {"durable": True},
        {"host_scope": "host"},
        {"streaming": True},
        {"listing": True},
        {"online_rebuild": True},
    ],
)
def test_impossible_minimum_capability_rejects_before_payload_staging(
    minimum: dict[str, object]
) -> None:
    composition = _composition()
    payload = _Payload()

    with pytest.raises(composition.CapabilityRequirementError):
        composition.StoreTopology(
            payload=payload, authority=_Authority(), minimum_capabilities=minimum
        ).resolve()

    assert payload.stage_calls == 0


def test_projection_can_never_satisfy_authority_capability_minima() -> None:
    composition = _composition()

    with pytest.raises(composition.CapabilityRequirementError):
        composition.StoreTopology(
            payload=_Payload(), authority=_Projection(), minimum_capabilities={"portable_query": True}
        ).resolve()


def test_unmet_minimum_rejects_named_factory_before_it_can_perform_io() -> None:
    composition = _composition()
    factory_calls: list[object] = []

    def payload_factory() -> _Payload:
        factory_calls.append(object())
        return _Payload()

    registry = composition.RoleRegistry()
    registry.register(
        "payload",
        "side-effecting",
        payload_factory,
        capabilities={
            "durable": False,
            "process_scope": "process",
            "host_scope": "process",
            "immutable_generations": True,
        },
    )

    with pytest.raises(composition.CapabilityRequirementError):
        composition.StoreTopology(
            payload=composition.BackendRef(name="side-effecting"),
            authority=_Authority(),
            minimum_capabilities={"durable": True},
        ).resolve(registry)

    assert factory_calls == []


def test_sqlite_local_scope_allows_conflict_or_retryable_timeout_not_universal_success() -> None:
    composition = _composition()

    outcomes = composition.allowed_progress_outcomes(authority_kind="sqlite-local")

    assert outcomes == {"success", "conflict", "retryable_timeout"}
