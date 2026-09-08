"""RoleRegistry contracts for payload participant construction.

The former global blob-backend registry was a development-only selector. This
module deliberately keeps its stable path while exercising the one per-topology
``RoleRegistry`` surface instead of recreating process-global mutable state.
"""

from __future__ import annotations

import pytest

from cacheness.storage.backends.blob_backends import (
    FilesystemBlobBackend,
    InMemoryBlobBackend,
)
from cacheness.storage.composition import (
    BackendRole,
    CompositionValidationError,
    RoleRegistration,
    RoleRegistry,
)


class RecordingPayload(InMemoryBlobBackend):
    """Payload factory used to prove exact construction-option forwarding."""

    def __init__(self, label: str, retry_limit: int = 0) -> None:
        super().__init__()
        self.label = label
        self.retry_limit = retry_limit


class RecordingProjection:
    """Minimal independent factory used for role/name collision coverage."""

    def __init__(self, label: str) -> None:
        self.label = label


def _wrong_signature(*, required: str) -> object:
    return object()


def test_fresh_registries_have_per_instance_builtin_registrations(tmp_path):
    """Built-ins use the same isolated registry path as application factories."""
    first = RoleRegistry()
    second = RoleRegistry()

    first_memory = first.construct(BackendRole.PAYLOAD, "memory", {})
    second_memory = second.construct(BackendRole.PAYLOAD, "memory", {})
    filesystem = first.construct(
        BackendRole.PAYLOAD,
        "filesystem",
        {"base_dir": tmp_path / "payloads", "shard_chars": 0},
    )

    assert isinstance(first_memory, InMemoryBlobBackend)
    assert isinstance(second_memory, InMemoryBlobBackend)
    assert first_memory is not second_memory
    assert isinstance(filesystem, FilesystemBlobBackend)
    assert filesystem.shard_chars == 0
    assert filesystem.base_dir == (tmp_path / "payloads").resolve()


def test_role_and_name_collisions_are_typed_and_replace_is_explicit():
    """A name is unique only within its typed role and replacement is opt-in."""
    registry = RoleRegistry()
    registry.register(BackendRole.PAYLOAD, "recording", RecordingPayload)
    registry.register(BackendRole.PROJECTION, "recording", RecordingProjection)

    with pytest.raises(CompositionValidationError, match="already registered"):
        registry.register(BackendRole.PAYLOAD, "recording", RecordingPayload)

    registry.register(
        BackendRole.PAYLOAD,
        "recording",
        RecordingPayload,
        replace=True,
    )

    assert registry.resolve(BackendRole.PAYLOAD, "recording").role == "payload"
    assert registry.resolve(BackendRole.PROJECTION, "recording").role == "projection"


@pytest.mark.parametrize("name", ["", 1, None])
def test_registration_name_validation_is_preserved(name):
    registry = RoleRegistry()

    with pytest.raises(CompositionValidationError, match="name must be a string"):
        registry.register(BackendRole.PAYLOAD, name, RecordingPayload)


def test_registration_rejects_non_callable_factories_and_invalid_roles():
    registry = RoleRegistry()

    with pytest.raises(CompositionValidationError, match="factory must be callable"):
        registry.register(BackendRole.PAYLOAD, "not-callable", object())
    with pytest.raises(CompositionValidationError, match="Unsupported participant role"):
        registry.register("authority-ish", "invalid", RecordingPayload)


def test_resolve_and_construct_translate_unknown_names_and_factory_errors():
    registry = RoleRegistry()
    registry.register(BackendRole.PAYLOAD, "wrong-signature", _wrong_signature)

    with pytest.raises(CompositionValidationError, match="No payload participant"):
        registry.resolve(BackendRole.PAYLOAD, "unknown")
    with pytest.raises(CompositionValidationError, match="Unable to construct"):
        registry.construct(BackendRole.PAYLOAD, "wrong-signature", {})


def test_role_registration_rejects_none_factory_results():
    registration = RoleRegistration(
        role=BackendRole.PAYLOAD.value,
        name="none",
        factory=lambda: None,
    )

    with pytest.raises(CompositionValidationError, match="returned None"):
        registration.construct({})


def test_registered_factory_receives_exact_options_and_constructs_fresh_instances():
    registry = RoleRegistry()
    registry.register(BackendRole.PAYLOAD, "recording", RecordingPayload)

    first = registry.construct(
        BackendRole.PAYLOAD,
        "recording",
        {"label": "primary", "retry_limit": 3},
    )
    second = registry.resolve(BackendRole.PAYLOAD, "recording").construct(
        {"label": "secondary", "retry_limit": 1}
    )

    assert isinstance(first, RecordingPayload)
    assert first.label == "primary"
    assert first.retry_limit == 3
    assert isinstance(second, RecordingPayload)
    assert second.label == "secondary"
    assert second.retry_limit == 1
    assert first is not second
