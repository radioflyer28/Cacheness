"""Role-aware participant registry contracts replacing metadata registries."""

from __future__ import annotations

import pytest

from cacheness.storage.composition import (
    BackendRole,
    CompositionValidationError,
    ProjectionSink,
    RoleRegistry,
)


class _RecordingProjection:
    """Minimal derived sink used to prove role-specific construction."""

    projection_name = "recording"

    def __init__(self, *, label: str = "recording") -> None:
        self.label = label
        self.applied: list[object] = []
        self.checkpoint: object | None = None

    def apply_projection_batch(self, batch: object) -> None:
        self.applied.append(batch)

    def save_projection_checkpoint(self, checkpoint: object) -> None:
        self.checkpoint = checkpoint

    def load_projection_checkpoint(self) -> object | None:
        return self.checkpoint


def test_registry_constructs_an_application_projection_under_one_shared_role() -> None:
    """Custom read models register as projections instead of metadata authorities."""
    registry = RoleRegistry()
    registry.register(
        BackendRole.PROJECTION,
        "recording",
        _RecordingProjection,
        capabilities={"projection_refresh": True},
    )

    projection = registry.construct(
        BackendRole.PROJECTION, "recording", {"label": "audit"}
    )

    assert isinstance(projection, ProjectionSink)
    assert projection.label == "audit"
    assert registry.resolve(BackendRole.PROJECTION, "recording").role == "projection"
    assert registry.capabilities(
        BackendRole.PROJECTION, "recording"
    ).projection_refresh is True


def test_registry_rejects_duplicate_names_within_the_same_role() -> None:
    """A derived participant cannot silently replace another derived participant."""
    registry = RoleRegistry()
    registry.register(BackendRole.PROJECTION, "recording", _RecordingProjection)

    with pytest.raises(CompositionValidationError, match="already registered"):
        registry.register(BackendRole.PROJECTION, "recording", _RecordingProjection)


def test_postgresql_name_cannot_be_resolved_as_a_lifecycle_authority() -> None:
    """A PostgreSQL projection is never a hidden canonical metadata selector."""
    registry = RoleRegistry()

    with pytest.raises(CompositionValidationError, match="No authority participant"):
        registry.resolve(BackendRole.AUTHORITY, "postgresql")
