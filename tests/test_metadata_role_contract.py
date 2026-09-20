"""Red contracts for authority/projection roles across metadata families."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import pytest


COMPOSITION_MODULE = "cacheness.storage.composition"


def _composition():
    assert find_spec(COMPOSITION_MODULE) is not None, (
        "Phase 4 must provide one explicit authority/projection role vocabulary"
    )
    return import_module(COMPOSITION_MODULE)


@pytest.mark.parametrize(
    ("implementation", "expected_role"),
    [
        ("memory", "authority"),
        ("sqlite", "authority"),
        ("json", "projection"),
        ("postgresql", "authority"),
    ],
)
def test_metadata_family_role_is_explicit(implementation: str, expected_role: str) -> None:
    composition = _composition()

    role = composition.resolve_metadata_role(implementation)

    assert role.kind == expected_role


@pytest.mark.parametrize("operation", ["read", "query_complete", "delete", "cleanup", "reconcile", "repair"])
def test_projection_role_cannot_authorize_lifecycle_or_query_completeness(operation: str) -> None:
    composition = _composition()
    projection = composition.resolve_metadata_role("json")

    assert projection.authorizes(operation) is False


def test_authority_role_exposes_only_narrow_catalog_transaction_primitives() -> None:
    composition = _composition()
    authority = composition.resolve_metadata_role("sqlite")

    assert authority.authorizes("query_complete") is True
    assert authority.authorizes("promote_catalog") is True
    assert authority.authorizes("payload_stage") is False
    assert authority.authorizes("payload_cleanup") is False


def test_role_registry_resolves_builtins_and_registered_implementations_identically(tmp_path) -> None:
    composition = _composition()
    registry = composition.RoleRegistry()
    registry.register("projection", "recording", lambda: {"role": "projection"})

    registered = registry.resolve("projection", "recording")
    builtin = registry.resolve("projection", "json")

    assert registered.role == builtin.role == "projection"
    sink = builtin.construct({"metadata_file": tmp_path / "derived.json"})
    assert isinstance(sink, composition.ProjectionSink)
    assert registry.capabilities("projection", "json").projection_refresh is True
    assert registry.capabilities("projection", "json").projection_rebuild is False


def test_duplicate_metadata_abc_and_factory_inheritance_are_not_a_role_contract() -> None:
    composition = _composition()

    with pytest.raises(composition.CompositionValidationError):
        composition.resolve_metadata_role("legacy-metadata-abc")
