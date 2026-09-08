"""Driver-boundary contracts for the PostgreSQL lifecycle authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cacheness.error_handling import CacheBlobMigrationRequiredError


@dataclass
class _Cursor:
    """Small DB-API transcript cursor used without a PostgreSQL service."""

    connection: "_Connection"
    row: tuple[Any, ...] | None = None
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    rowcount: int = 1

    def execute(self, query: object, params: object = None) -> "_Cursor":
        self.connection.executions.append((query, params))
        response = self.connection.responses.pop(0) if self.connection.responses else None
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            self.row = response
        elif isinstance(response, list):
            self.rows = response
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.row

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def close(self) -> None:
        return None


@dataclass
class _Transaction:
    connection: "_Connection"

    def __enter__(self) -> "_Transaction":
        self.connection.transaction_count += 1
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is not None:
            self.connection.rollback_count += 1
        else:
            self.connection.commit_count += 1
        return False


@dataclass
class _Connection:
    responses: list[object] = field(default_factory=list)
    executions: list[tuple[object, object]] = field(default_factory=list)
    transaction_count: int = 0
    commit_count: int = 0
    rollback_count: int = 0
    closed: bool = False

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    def close(self) -> None:
        self.closed = True


@dataclass
class _Factory:
    connections: list[_Connection] = field(default_factory=list)
    calls: int = 0

    def __call__(self) -> _Connection:
        self.calls += 1
        connection = _Connection()
        self.connections.append(connection)
        return connection


def _query_text(query: object) -> str:
    """Normalize recording-driver query objects without a live connection."""
    return str(query).lower()


def test_constructor_is_non_materializing_and_initialize_is_explicit() -> None:
    """PostgreSQL DDL is available only through the explicit initialize boundary."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    factory = _Factory()
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    assert factory.calls == 0
    authority.initialize()

    assert factory.calls == 1
    statements = [_query_text(query) for query, _ in factory.connections[0].executions]
    assert any("create schema" in statement for statement in statements)
    assert any("create table" in statement for statement in statements)
    assert factory.connections[0].commit_count == 1
    assert factory.connections[0].closed is True


def test_reopen_rejects_wrong_version_without_ddl_or_mutation() -> None:
    """A foreign or future layout is a Phase 7 migration/rebuild boundary."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    factory = _Factory()
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")
    # The read-only validation query finds the owned marker but an unsupported
    # schema version.  It must fail before any mutating statement is sent.
    factory.connections.append(_Connection(responses=[(999, "identity", "postgresql-authority-v1")]))

    with pytest.raises(CacheBlobMigrationRequiredError):
        authority.open()

    statements = [_query_text(query) for query, _ in factory.connections[-1].executions]
    assert not any("create " in statement or "alter " in statement for statement in statements)
    assert factory.connections[-1].commit_count == 0
    assert factory.connections[-1].closed is True


def test_schema_identifier_is_validated_before_driver_use() -> None:
    """Schema names are identifiers, never interpolated value strings."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    factory = _Factory()
    with pytest.raises(ValueError, match="schema"):
        PostgresqlLifecycleAuthority(factory, schema="authority; drop schema public")
    assert factory.calls == 0
