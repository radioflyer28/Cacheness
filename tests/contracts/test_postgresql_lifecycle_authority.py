"""Driver-boundary contracts for the PostgreSQL lifecycle authority."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import pytest

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationRequiredError,
)
from cacheness.storage.lifecycle_authority import (
    EntryExpectation,
    MutationSpec,
    PreparedMutation,
    VerificationProof,
)


@dataclass
class _Cursor:
    """Small DB-API transcript cursor used without a PostgreSQL service."""

    connection: "_Connection"
    row: tuple[Any, ...] | None = None
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    rowcount: int = 1

    def execute(self, query: object, params: object = None) -> "_Cursor":
        self.connection.executions.append((query, params))
        if "set local" in _query_text(query):
            return self
        self.row = None
        self.rows = []
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
            if self.connection.commit_error is not None:
                raise self.connection.commit_error
        return False


@dataclass
class _Connection:
    responses: list[object] = field(default_factory=list)
    executions: list[tuple[object, object]] = field(default_factory=list)
    transaction_count: int = 0
    commit_count: int = 0
    rollback_count: int = 0
    closed: bool = False
    commit_error: BaseException | None = None

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    def close(self) -> None:
        self.closed = True


@dataclass
class _Factory:
    scripts: list[list[object]] = field(default_factory=list)
    commit_errors: list[BaseException | None] = field(default_factory=list)
    connections: list[_Connection] = field(default_factory=list)
    calls: int = 0

    def __call__(self) -> _Connection:
        self.calls += 1
        responses = self.scripts.pop(0) if self.scripts else []
        commit_error = self.commit_errors.pop(0) if self.commit_errors else None
        connection = _Connection(responses=responses, commit_error=commit_error)
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

    factory = _Factory(
        scripts=[[(999, "identity", "postgresql-lifecycle-authority-v1")]]
    )
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")
    # The read-only validation query finds the owned marker but an unsupported
    # schema version.  It must fail before any mutating statement is sent.
    with pytest.raises(CacheBlobMigrationRequiredError):
        authority.open()

    statements = [_query_text(query) for query, _ in factory.connections[-1].executions]
    assert not any("create " in statement or "alter " in statement for statement in statements)
    assert factory.connections[-1].commit_count == 0
    assert factory.connections[-1].closed is True


def test_open_validates_exact_layout_without_emitting_ddl() -> None:
    """A reopened current layout is inspected read-only, not silently upgraded."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        POSTGRESQL_AUTHORITY_CAPABILITY,
        POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
        PostgresqlLifecycleAuthority,
    )

    required_tables = [
        ("authority_meta",),
        ("entry_lineage",),
        ("entries",),
        ("mutations",),
        ("cleanup_debt",),
        ("clear_runs",),
        ("clear_targets",),
        ("reconciliation_runs",),
        ("reconciliation_actions",),
    ]
    required_constraints = [
        ("authority_meta_singleton_check",),
        ("mutations_operation_id_key",),
        ("cleanup_debt_operation_locator_role_key",),
        ("clear_targets_run_id_key_key",),
        ("reconciliation_actions_run_id_action_id_key",),
    ]
    factory = _Factory(
        scripts=[[
            (POSTGRESQL_AUTHORITY_SCHEMA_VERSION, "identity", POSTGRESQL_AUTHORITY_CAPABILITY),
            required_tables,
            required_constraints,
        ]]
    )

    PostgresqlLifecycleAuthority(factory, schema="phase5_authority").open()

    statements = [_query_text(query) for query, _ in factory.connections[0].executions]
    assert not any("create " in statement or "alter " in statement for statement in statements)
    assert factory.connections[0].rollback_count == 0


def test_initialize_rolls_back_and_redacts_driver_details() -> None:
    """Driver failures preserve a cause without exposing a PostgreSQL DSN."""
    from cacheness.error_handling import CacheBlobBackendError
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    secret = "postgresql://user:super-secret@example.invalid/cacheness"
    factory = _Factory(scripts=[[RuntimeError(secret)]])
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    with pytest.raises(CacheBlobBackendError) as captured:
        authority.initialize()

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value.context)
    assert factory.connections[0].rollback_count == 1


def test_initialize_rejects_an_existing_partial_layout_before_ddl() -> None:
    """Explicit initialization never adopts or completes a foreign partial schema."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    factory = _Factory(scripts=[[[("entries",)], None]])
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    with pytest.raises(CacheBlobMigrationRequiredError):
        authority.initialize()

    statements = [_query_text(query) for query, _ in factory.connections[0].executions]
    assert not any("create " in statement or "alter " in statement for statement in statements)
    assert factory.connections[0].rollback_count == 1


def test_schema_identifier_is_validated_before_driver_use() -> None:
    """Schema names are identifiers, never interpolated value strings."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    factory = _Factory()
    with pytest.raises(ValueError, match="schema"):
        PostgresqlLifecycleAuthority(factory, schema="authority; drop schema public")
    assert factory.calls == 0


def _spec(operation_id: str = "operation-1") -> MutationSpec:
    """Build one transition with a fully bounded, canonical descriptor."""
    return MutationSpec.create(
        operation_id=operation_id,
        key="authority-key",
        generation="generation-1",
        candidate_locator="generations/generation-1.native",
        expected=EntryExpectation.absent(),
        manifest=b"canonical-manifest",
    )


def test_prepare_rejects_a_stale_expectation_without_creating_intent() -> None:
    """A stale exact expectation is a conflict, never an implicit retry."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    factory = _Factory(scripts=[[None, (3, 7, "old-generation", "a" * 64)]])
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    with pytest.raises(CacheBlobLifecycleConflictError):
        authority.prepare_mutation(_spec())

    statements = [_query_text(query) for query, _ in factory.connections[0].executions]
    assert not any("insert into" in statement and "mutations" in statement for statement in statements)
    assert factory.connections[0].rollback_count == 1


def test_promotion_marks_fresh_lineage_before_matching_absence() -> None:
    """A fresh lineage sentinel must not turn its own absent create into ABA."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    spec = _spec()
    prepared = PreparedMutation(spec.operation_id, spec)
    digest = hashlib.sha256(spec.manifest).hexdigest()
    mutation_row = (
        spec.key,
        spec.generation,
        spec.candidate_locator,
        None,
        None,
        None,
        None,
        spec.manifest,
        digest,
        len(spec.manifest),
        "prepared",
    )
    factory = _Factory(
        scripts=[[
            mutation_row,
            (spec.key,),
            (0, None, None, None),
            None,
            (0,),
            (1,),
            (spec.generation, spec.candidate_locator),
            (spec.operation_id,),
            (1,),
            (
                spec.key,
                spec.generation,
                spec.candidate_locator,
                spec.manifest,
                digest,
                1,
                1,
            ),
            [],
        ]]
    )

    PostgresqlLifecycleAuthority(factory, schema="phase5_authority").promote_mutation(
        prepared
    )

    statements = [_query_text(query) for query, _ in factory.connections[0].executions]
    lineage_insert = next(statement for statement in statements if "entry_lineage" in statement)
    assert "returning key" in lineage_insert


def test_prepare_and_verification_use_exact_bound_values() -> None:
    """Intent replay is operation-idempotent and proof writes are exact CAS updates."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    spec = _spec()
    factory = _Factory(scripts=[[None, None, None], [(spec.operation_id,)]])
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    prepared = authority.prepare_mutation(spec)
    authority.record_verification(
        prepared, VerificationProof("c" * 64, len(spec.manifest), spec.manifest)
    )

    executions = [
        execution
        for connection in factory.connections
        for execution in connection.executions
    ]
    insert_query, insert_params = next(
        (query, params)
        for query, params in executions
        if "insert into" in _query_text(query) and "mutations" in _query_text(query)
    )
    assert "%s" in _query_text(insert_query)
    assert spec.operation_id in insert_params
    verification_query = next(
        query
        for query, _ in executions
        if "update" in _query_text(query) and "verified_digest" in _query_text(query)
    )
    assert "returning operation_id" in _query_text(verification_query)


def test_verification_rejects_zero_affected_rows_and_rolls_back() -> None:
    """A proof can only attach to its exact prepared mutation."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    spec = _spec()
    factory = _Factory(scripts=[[None]])
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    with pytest.raises(CacheBlobLifecycleConflictError):
        authority.record_verification(
            PreparedMutation(spec.operation_id, spec),
            VerificationProof("d" * 64, 0, spec.manifest),
        )

    assert factory.connections[0].rollback_count == 1


def test_abort_records_candidate_debt_without_payload_effects() -> None:
    """Abort retains exact candidate cleanup debt but never performs object I/O."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    spec = _spec()
    mutation_row = (
        spec.key,
        spec.generation,
        spec.candidate_locator,
        None,
        None,
        None,
        None,
        spec.manifest,
        None,
        None,
        "prepared",
    )
    factory = _Factory(scripts=[[mutation_row, (spec.operation_id,), None]])
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    authority.abort_mutation(PreparedMutation(spec.operation_id, spec), candidate_persisted=True)

    statements = [_query_text(query) for query, _ in factory.connections[0].executions]
    assert any("cleanup_debt" in statement and "insert into" in statement for statement in statements)
    assert not any("boto3" in statement or "s3" in statement for statement in statements)


def test_uncertain_promotion_reopens_a_fresh_lease_for_exact_operation_state() -> None:
    """A commit transport failure is classified from durable operation identity, never guessed."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    spec = _spec()
    prepared = PreparedMutation(spec.operation_id, spec)
    digest = hashlib.sha256(spec.manifest).hexdigest()
    prepared_row = (
        spec.key, spec.generation, spec.candidate_locator, None, None, None, None,
        spec.manifest, digest, len(spec.manifest), "prepared",
    )
    promoted_row = prepared_row[:-1] + ("promoted",)
    entry_row = (
        spec.key, spec.generation, spec.candidate_locator, spec.manifest, digest, 1, 1,
    )
    factory = _Factory(
        scripts=[[
            prepared_row, (spec.key,), (0, None, None, None), None, (0,), (1,),
            (spec.generation, spec.candidate_locator), (spec.operation_id,), (1,),
            entry_row, [],
        ], [promoted_row, entry_row, []]],
        commit_errors=[RuntimeError("connection closed during commit"), None],
    )

    result = PostgresqlLifecycleAuthority(factory, schema="phase5_authority").promote_mutation(
        prepared
    )

    assert result.entry.generation == spec.generation
    assert factory.calls == 2
