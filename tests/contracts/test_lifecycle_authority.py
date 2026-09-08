"""Tier-aware semantic contracts for lifecycle-authority adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
)
from cacheness.storage.lifecycle_authority import (
    EntryExpectation,
    MutationSpec,
    VerificationProof,
)
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


@dataclass
class _Cursor:
    """One response-driven DB-API cursor for PostgreSQL boundary tests."""

    connection: "_Connection"
    row: tuple[Any, ...] | None = None
    rows: list[tuple[Any, ...]] = field(default_factory=list)

    def execute(self, query: object, params: object = None) -> "_Cursor":
        del params
        if "set local" in str(query).lower():
            return self
        self.row = None
        self.rows = []
        response = self.connection.responses.pop(0)
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
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc, traceback
        self.connection.rolled_back = exc_type is not None
        return False


@dataclass
class _Connection:
    responses: list[object]
    rolled_back: bool = False

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    def close(self) -> None:
        return None


class _SqlstateError(RuntimeError):
    """Minimal driver error carrying the PostgreSQL SQLSTATE boundary."""

    def __init__(self, message: str, sqlstate: str) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


def _spec() -> MutationSpec:
    return MutationSpec.create(
        operation_id="progress-operation",
        key="progress-key",
        generation="generation-1",
        candidate_locator="generations/generation-1.native",
        expected=EntryExpectation.absent(),
        manifest=b"manifest",
    )


def _promote(authority: object, operation_id: str) -> object:
    prepared = authority.prepare_mutation(
        MutationSpec.create(
            operation_id=operation_id,
            key="tier-key",
            generation=f"generation-{operation_id}",
            candidate_locator=f"generations/{operation_id}.native",
            expected=EntryExpectation.absent(),
            manifest=b"tier-manifest",
        )
    )
    authority.record_verification(prepared, VerificationProof("a" * 64, 1))
    return authority.promote_mutation(prepared)


@pytest.mark.parametrize("kind", ("memory", "sqlite"))
def test_local_tiers_share_safety_without_claiming_equal_progress(
    tmp_path: Path, kind: str
) -> None:
    """A completed generation is safe, while each topology publishes its own progress tier."""
    authority = (
        InMemoryLifecycleAuthority()
        if kind == "memory"
        else SqliteLifecycleAuthority.for_root(tmp_path / "sqlite-tier")
    )
    promoted = _promote(authority, kind)

    assert authority.read_entry("tier-key") == promoted.entry
    with pytest.raises(CacheBlobLifecycleConflictError):
        authority.delete_entry("tier-key", expected=EntryExpectation.absent())
    authority.close()


@pytest.mark.parametrize(
    ("sqlstate", "outcome"),
    [
        ("40001", "serialization"),
        ("40P01", "deadlock"),
        ("55P03", "lock_timeout"),
        ("57014", "statement_timeout"),
    ],
)
def test_postgresql_sqlstate_progress_is_typed_bounded_and_causal(
    sqlstate: str, outcome: str
) -> None:
    """Retryable PostgreSQL progress states never masquerade as conflicts or misses."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    cause = _SqlstateError("driver detail must stay causal", sqlstate)
    authority = PostgresqlLifecycleAuthority(
        lambda: _Connection([None, cause]), schema="phase5_authority"
    )

    assert PostgresqlLifecycleAuthority.progress_outcome_for_sqlstate(sqlstate) == outcome
    with pytest.raises(CacheBlobLifecycleTimeoutError) as captured:
        authority.prepare_mutation(_spec())

    assert captured.value.__cause__ is cause
    assert captured.value.context == {
        "operation": "postgresql_lifecycle_authority",
        "stage": "prepare_mutation",
        "sqlstate": sqlstate,
        "progress_outcome": outcome,
    }


def test_postgresql_exact_conflict_and_fatal_driver_failure_stay_distinct() -> None:
    """CAS conflicts and fatal SQL errors remain distinct from retryable progress outcomes."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    conflict = PostgresqlLifecycleAuthority(
        lambda: _Connection([None, (1, 1, "old", "a" * 64)]),
        schema="phase5_authority",
    )
    with pytest.raises(CacheBlobLifecycleConflictError):
        conflict.prepare_mutation(_spec())

    fatal = _SqlstateError("syntax failure", "42601")
    backend = PostgresqlLifecycleAuthority(
        lambda: _Connection([None, fatal]), schema="phase5_authority"
    )
    with pytest.raises(CacheBlobBackendError) as captured:
        backend.prepare_mutation(_spec())
    assert captured.value.__cause__ is fatal
    assert "progress_outcome" not in captured.value.context
