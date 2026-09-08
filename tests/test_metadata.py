"""Metadata-facing role and error contracts after the topology cutover."""

from __future__ import annotations

import pytest

from cacheness.error_handling import CacheReason, CacheStorageError
from cacheness.storage.composition import (
    BackendRole,
    CompositionValidationError,
    resolve_metadata_role,
)


@pytest.mark.parametrize(
    ("implementation", "expected_role"),
    (
        ("memory", BackendRole.AUTHORITY),
        ("sqlite", BackendRole.AUTHORITY),
        ("json", BackendRole.PROJECTION),
        ("postgresql", BackendRole.PROJECTION),
    ),
)
def test_metadata_implementation_roles_are_explicit_and_backend_independent(
    implementation: str, expected_role: BackendRole
) -> None:
    """Metadata families declare authority or derived-projection status directly."""
    role = resolve_metadata_role(implementation)

    assert role.kind == expected_role.value
    assert role.authorizes("promote_catalog") is (expected_role is BackendRole.AUTHORITY)
    assert role.authorizes("query_complete") is (expected_role is BackendRole.AUTHORITY)


def test_unknown_metadata_implementation_fails_before_any_selection_io() -> None:
    """The replacement surface has no implicit metadata-backend fallback."""
    with pytest.raises(CompositionValidationError, match="has no Phase 4 role"):
        resolve_metadata_role("unknown")


def test_metadata_corruption_keeps_the_typed_reason_code_contract() -> None:
    """Callers can distinguish rejected metadata from an ordinary cache miss."""
    error = CacheStorageError(
        "metadata is corrupt", context={"reason": CacheReason.METADATA_CORRUPT.value}
    )

    assert error.context["reason"] == CacheReason.METADATA_CORRUPT.value
