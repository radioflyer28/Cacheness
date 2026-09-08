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


def test_postgresql_is_deferred_until_a_qualified_projection_sink_exists() -> None:
    """PostgreSQL remains classified, but is not advertised as constructible yet."""
    registry = RoleRegistry()

    with pytest.raises(CompositionValidationError, match="No projection participant"):
        registry.resolve(BackendRole.PROJECTION, "postgresql")
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
