"""PostgreSQL optional-dependency and derived-projection contracts."""

from __future__ import annotations

import pytest

from cacheness.storage.composition import (
    BackendRole,
    CompositionValidationError,
    RoleRegistry,
    resolve_metadata_role,
)


def test_postgresql_is_declared_as_a_derived_projection_not_an_authority() -> None:
    """PostgreSQL cannot grant canonical query or promotion authority."""
    role = resolve_metadata_role("postgresql")

    assert role.kind == BackendRole.PROJECTION.value
    assert role.authorizes("promote_catalog") is False
    assert role.authorizes("query_complete") is False


def test_postgresql_is_registered_only_in_the_projection_role() -> None:
    """The composition registry rejects the retired authority selection path."""
    registry = RoleRegistry()

    assert registry.resolve(BackendRole.PROJECTION, "postgresql").role == "projection"
    with pytest.raises(CompositionValidationError, match="No authority participant"):
        registry.resolve(BackendRole.AUTHORITY, "postgresql")


def test_postgresql_optional_dependency_failure_is_explicit() -> None:
    """An unavailable optional driver fails at construction with installation guidance."""
    from cacheness.storage.backends import postgresql_backend

    assert isinstance(postgresql_backend.SQLALCHEMY_AVAILABLE, bool)
    assert isinstance(postgresql_backend.PSYCOPG_AVAILABLE, bool)

    if not postgresql_backend.SQLALCHEMY_AVAILABLE:
        with pytest.raises(ImportError, match="SQLAlchemy"):
            postgresql_backend.PostgresBackend("postgresql://localhost/cacheness")
    elif not postgresql_backend.PSYCOPG_AVAILABLE:
        with pytest.raises(ImportError, match="PostgreSQL driver"):
            postgresql_backend.PostgresBackend("postgresql://localhost/cacheness")
