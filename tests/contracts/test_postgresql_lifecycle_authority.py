"""Driver-boundary contracts for the PostgreSQL lifecycle authority."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import pytest

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobMigrationOfflineDecisionRequiredError,
    CacheBlobMigrationRequiredError,
)
from cacheness.storage.lifecycle_authority import (
    CleanupDebt,
    EntryExpectation,
    PageToken,
    MutationSpec,
    PreparedMutation,
    ProjectionRevision,
    VerificationProof,
)
from cacheness.storage.migration_authority import (
    AuthorityIdentitySnapshot,
    AuthorityInventoryEntry,
    AuthorityPublicationState,
    VerifiedCandidateReceipt,
    candidate_digest,
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


@dataclass
class _Lease:
    """Pool-like connection lease that records the exception passed to release."""

    connection: _Connection
    exit_type: type[BaseException] | None = None

    def __enter__(self) -> _Connection:
        return self.connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        self.exit_type = exc_type
        self.connection.close()
        return False


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


def test_blob_store_public_initialize_provisions_a_fresh_postgresql_authority(
    tmp_path,
) -> None:
    """The public remote boundary creates before it performs read-only validation."""
    from cacheness.storage import BlobStore
    from cacheness.storage.backends.blob_backends import InMemoryBlobBackend
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        POSTGRESQL_AUTHORITY_CAPABILITY,
        POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
        PostgresqlLifecycleAuthority,
    )
    from cacheness.storage.composition import BackendRef, StoreTopology

    class StaticManifestKey:
        def get_key(self) -> bytes:
            return b"r" * 32

    class RemotePayload(InMemoryBlobBackend):
        qualification_identity = "s3"
        topology_capabilities = {
            "durable": True,
            "process_scope": "multi_host",
            "host_scope": "multi_host",
            "immutable_generations": True,
            "streaming": True,
            "listing": True,
        }

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
        ("migration_store_entries",),
    ]
    required_constraints = [
        ("authority_meta_singleton_check",),
        ("mutations_operation_id_key",),
        ("mutations_mutation_id_key",),
        ("cleanup_debt_operation_locator_role_key",),
        ("clear_targets_run_id_key_key",),
        ("reconciliation_actions_run_id_source_action_id_key",),
        ("migration_store_entries_run_id_selection_key_key",),
    ]
    factory = _Factory(
        scripts=[
            [],
            [
                (
                    POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
                    "fresh-store",
                    POSTGRESQL_AUTHORITY_CAPABILITY,
                ),
                required_tables,
                required_constraints,
            ],
        ]
    )
    authority = PostgresqlLifecycleAuthority(
        factory, schema="phase5_authority", store_identity="fresh-store"
    )
    authority.qualification_identity = "postgresql"
    payload = RemotePayload()
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(instance=payload),
            authority=BackendRef(instance=authority),
        ),
        cache_dir=tmp_path / "remote-store",
        manifest_key_provider=StaticManifestKey(),
    )
    try:
        store.initialize()
        assert factory.calls == 2
        first = [_query_text(query) for query, _ in factory.connections[0].executions]
        second = [_query_text(query) for query, _ in factory.connections[1].executions]
        assert any("create schema" in statement for statement in first)
        assert not any("create " in statement for statement in second)
    finally:
        store.close()
        authority.close()
        payload.close()


def test_fresh_blobstore_initialize_rechecks_activated_offline_after_postgresql_identity_load(
    tmp_path,
) -> None:
    """A fresh remote worker must fence persisted offline activation before readiness."""
    from cacheness.storage import BlobStore
    from cacheness.storage.backends.blob_backends import InMemoryBlobBackend
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        POSTGRESQL_AUTHORITY_CAPABILITY,
        POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
        PostgresqlLifecycleAuthority,
    )
    from cacheness.storage.composition import BackendRef, StoreTopology

    events: list[str] = []

    class RecordingManifestKey:
        def get_key(self) -> bytes:
            events.append("manifest_key")
            return b"r" * 32

    class RecordingRemotePayload(InMemoryBlobBackend):
        qualification_identity = "s3"
        topology_capabilities = {
            "durable": True,
            "process_scope": "multi_host",
            "host_scope": "multi_host",
            "immutable_generations": True,
            "streaming": True,
            "listing": True,
        }

        def materialize_handler_io(self) -> object:
            events.append("materialize")
            return super().materialize_handler_io()

    class RecordingAuthority(PostgresqlLifecycleAuthority):
        qualification_identity = "postgresql"

        def initialize(self) -> None:
            events.append("authority_initialize")
            super().initialize()

        def publication_state(self) -> AuthorityPublicationState:
            events.append(f"publication_state:{self.store_identity}")
            return super().publication_state()

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
        ("migration_store_entries",),
    ]
    required_constraints = [
        ("authority_meta_singleton_check",),
        ("mutations_operation_id_key",),
        ("mutations_mutation_id_key",),
        ("cleanup_debt_operation_locator_role_key",),
        ("clear_targets_run_id_key_key",),
        ("reconciliation_actions_run_id_source_action_id_key",),
        ("migration_store_entries_run_id_selection_key_key",),
    ]
    factory = _Factory(
        scripts=[
            [],
            [
                (
                    POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
                    "persisted-store",
                    POSTGRESQL_AUTHORITY_CAPABILITY,
                ),
                required_tables,
                required_constraints,
            ],
            [
                (
                    7,
                    "run-remote-cutover",
                    "a" * 64,
                    "b" * 64,
                    7,
                    8,
                    AuthorityPublicationState.ACTIVATED_OFFLINE.value,
                    "candidate",
                    True,
                )
            ],
        ]
    )
    authority = RecordingAuthority(factory, schema="phase7_authority")
    payload = RecordingRemotePayload()
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(instance=payload),
            authority=BackendRef(instance=authority),
        ),
        cache_dir=tmp_path / "remote-store",
        manifest_key_provider=RecordingManifestKey(),
    )

    try:
        with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError):
            store.initialize()

        assert authority.store_identity == "persisted-store"
        assert events == [
            "publication_state:None",
            "authority_initialize",
            "publication_state:persisted-store",
        ]
        assert factory.calls == 3
        assert store._initialized is False
        assert store.guarded_handler_io is None
    finally:
        store.close()
        authority.close()
        payload.close()


def test_postgresql_preflight_mutation_checks_persisted_worker_fence_after_open() -> None:
    """Preflight reloads remote authority state before admitting an ordinary worker."""
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
        ("migration_store_entries",),
    ]
    required_constraints = [
        ("authority_meta_singleton_check",),
        ("mutations_operation_id_key",),
        ("mutations_mutation_id_key",),
        ("cleanup_debt_operation_locator_role_key",),
        ("clear_targets_run_id_key_key",),
        ("reconciliation_actions_run_id_source_action_id_key",),
        ("migration_store_entries_run_id_selection_key_key",),
    ]
    factory = _Factory(
        scripts=[
            [
                (
                    POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
                    "persisted-store",
                    POSTGRESQL_AUTHORITY_CAPABILITY,
                ),
                required_tables,
                required_constraints,
            ],
            [
                (
                    7,
                    "run-remote-cutover",
                    "a" * 64,
                    "b" * 64,
                    7,
                    8,
                    AuthorityPublicationState.ACTIVATED_OFFLINE.value,
                    "candidate",
                    True,
                )
            ],
        ]
    )
    authority = PostgresqlLifecycleAuthority(factory, schema="phase7_authority")

    with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError):
        authority.preflight_mutation()

    assert authority.store_identity == "persisted-store"
    assert factory.calls == 2


@pytest.mark.parametrize(
    ("state_row", "expected_error"),
    [
        pytest.param(
            (7, None, None, None, None, None, "idle", "source", False),
            None,
            id="idle",
        ),
        pytest.param(
            (8, "run-remote-cutover", "a" * 64, "b" * 64, 7, 8, "active", "candidate", False),
            None,
            id="active",
        ),
        pytest.param(
            ("invalid", None, None, None, None, None, "idle", "source", False),
            CacheBlobMigrationRequiredError,
            id="malformed",
        ),
    ],
)
def test_postgresql_preflight_mutation_preserves_worker_states_and_fails_closed(
    state_row: tuple[object, ...],
    expected_error: type[Exception] | None,
) -> None:
    """Preflight permits persisted ordinary states and preserves typed corruption errors."""
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
        ("migration_store_entries",),
    ]
    required_constraints = [
        ("authority_meta_singleton_check",),
        ("mutations_operation_id_key",),
        ("mutations_mutation_id_key",),
        ("cleanup_debt_operation_locator_role_key",),
        ("clear_targets_run_id_key_key",),
        ("reconciliation_actions_run_id_source_action_id_key",),
        ("migration_store_entries_run_id_selection_key_key",),
    ]
    factory = _Factory(
        scripts=[
            [
                (
                    POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
                    "persisted-store",
                    POSTGRESQL_AUTHORITY_CAPABILITY,
                ),
                required_tables,
                required_constraints,
            ],
            [state_row],
        ]
    )
    authority = PostgresqlLifecycleAuthority(factory, schema="phase7_authority")

    if expected_error is None:
        authority.preflight_mutation()
    else:
        with pytest.raises(expected_error):
            authority.preflight_mutation()

    assert authority.store_identity == "persisted-store"
    assert factory.calls == 2


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
        ("migration_store_entries",),
    ]
    required_constraints = [
        ("authority_meta_singleton_check",),
        ("mutations_operation_id_key",),
        ("mutations_mutation_id_key",),
        ("cleanup_debt_operation_locator_role_key",),
        ("clear_targets_run_id_key_key",),
        ("reconciliation_actions_run_id_source_action_id_key",),
        ("migration_store_entries_run_id_selection_key_key",),
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


def test_pool_lease_receives_transition_failure_for_rollback() -> None:
    """Caller-owned pool leases observe failures instead of an unconditional success exit."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    connection = _Connection(responses=[None, (3, 7, "old-generation", "a" * 64)])
    lease = _Lease(connection)
    authority = PostgresqlLifecycleAuthority(lambda: lease, schema="phase5_authority")

    with pytest.raises(CacheBlobLifecycleConflictError):
        authority.prepare_mutation(_spec())

    assert lease.exit_type is CacheBlobLifecycleConflictError


def test_promotion_marks_fresh_lineage_before_matching_absence() -> None:
    """A fresh lineage sentinel must not turn its own absent create into ABA."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    spec = _spec()
    prepared = PreparedMutation(spec.operation_id, spec)
    payload_digest = hashlib.sha256(b"verified payload bytes").hexdigest()
    manifest_digest = hashlib.sha256(spec.manifest).hexdigest()
    mutation_row = (
        spec.key,
        spec.generation,
        spec.candidate_locator,
        None,
        None,
        None,
        None,
        spec.manifest,
        payload_digest,
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
                1,
                1,
            ),
            [],
        ]]
    )

    result = PostgresqlLifecycleAuthority(factory, schema="phase5_authority").promote_mutation(
        prepared
    )

    assert result.entry.expectation.manifest_digest == manifest_digest
    assert result.entry.expectation.manifest_digest != payload_digest

    statements = [_query_text(query) for query, _ in factory.connections[0].executions]
    lineage_insert = next(statement for statement in statements if "entry_lineage" in statement)
    assert "returning key" in lineage_insert
    promotion_update = next(
        statement
        for statement in statements
        if "update" in statement and "promoted_lineage" in statement
    )
    assert "promoted_revision" in promotion_update
    replay_query = next(
        statement
        for statement in statements
        if "select key, generation, locator, manifest, promoted_lineage" in statement
        and "state = 'promoted'" in statement
    )
    assert "join" not in replay_query


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
    payload_digest = hashlib.sha256(b"verified payload bytes").hexdigest()
    prepared_row = (
        spec.key, spec.generation, spec.candidate_locator, None, None, None, None,
        spec.manifest, payload_digest, len(spec.manifest), "prepared",
    )
    promoted_row = prepared_row[:-1] + ("promoted",)
    promotion_receipt_row = (
        spec.key, spec.generation, spec.candidate_locator, spec.manifest, 1, 1,
    )
    factory = _Factory(
        scripts=[[
            prepared_row, (spec.key,), (0, None, None, None), None, (0,), (1,),
            (spec.generation, spec.candidate_locator), (spec.operation_id,), (1,),
            promotion_receipt_row, [],
        ], [promoted_row, promotion_receipt_row, []]],
        commit_errors=[RuntimeError("connection closed during commit"), None],
    )

    result = PostgresqlLifecycleAuthority(factory, schema="phase5_authority").promote_mutation(
        prepared
    )

    assert result.entry.generation == spec.generation
    assert factory.calls == 2


def test_reconciliation_cursor_advances_only_to_emitted_byte_bounded_mutations() -> None:
    """Resumed pages retain prepared evidence that did not fit the prior byte bound."""
    from cacheness.config import LifecycleLimits
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )
    from cacheness.storage.lifecycle_authority import ReconciliationSnapshot

    def mutation_row(identifier: int) -> tuple[object, ...]:
        return (
            identifier,
            f"operation-{identifier}",
            f"key-{identifier}",
            f"generation-{identifier}",
            f"generations/{identifier}.native",
            None,
            None,
            None,
            None,
            b"abcdef",
            "prepared",
        )

    first, second, third = (mutation_row(identifier) for identifier in (1, 2, 3))
    factory = _Factory(
        scripts=[
            [[first, second, third], []],
            [[second, third], []],
            [[third], []],
        ]
    )
    authority = PostgresqlLifecycleAuthority(
        factory,
        schema="phase5_authority",
        lifecycle_limits=LifecycleLimits(
            operation_page_size=6,
            max_operation_record_bytes=11,
        ),
    )
    snapshot = ReconciliationSnapshot(0, 3, 0)

    page_one = authority.page_reconciliation_work(
        snapshot, mutation_cursor=0, debt_cursor=0
    )
    page_two = authority.page_reconciliation_work(
        snapshot, mutation_cursor=page_one.mutation_cursor, debt_cursor=0
    )
    page_three = authority.page_reconciliation_work(
        snapshot, mutation_cursor=page_two.mutation_cursor, debt_cursor=0
    )

    assert tuple(work.row_id for work in page_one.works) == (1,)
    assert tuple(work.row_id for work in page_two.works) == (2,)
    assert tuple(work.row_id for work in page_three.works) == (3,)
    assert (
        page_one.mutation_cursor,
        page_two.mutation_cursor,
        page_three.mutation_cursor,
    ) == (1, 2, 3)


def test_inventory_attribution_uses_one_snapshot_consistent_bounded_query() -> None:
    """An inventory page can prove only its own snapshot-owned locators."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )
    from cacheness.storage.lifecycle_authority import ReconciliationSnapshot

    factory = _Factory(
        scripts=[
            [[(7, "generations/committed.native")]],
            [[(8, None)]],
        ]
    )
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")
    snapshot = ReconciliationSnapshot(7, 11, 13)
    locators = ("generations/committed.native", "generations/unknown.native")

    assert authority.inventory_locator_attribution(snapshot, locators) == frozenset(
        {"generations/committed.native"}
    )
    assert authority.inventory_locator_attribution(snapshot, locators) is None

    query, params = factory.connections[0].executions[-1]
    assert "with revision" in _query_text(query)
    assert "left join owned" in _query_text(query)
    assert params == (list(locators), 11, list(locators), 13, list(locators))
    assert factory.connections[0].transaction_count == 1
    assert factory.connections[1].transaction_count == 1


def test_inventory_pages_raw_rows_at_one_postgresql_revision_without_live_claims() -> None:
    """Deterministic DB-API coverage only; Phase 8 owns live PostgreSQL and AWS S3 proof."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        POSTGRESQL_AUTHORITY_CAPABILITY,
        POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
        PostgresqlLifecycleAuthority,
    )

    first_manifest = b"first-opaque-manifest"
    second_manifest = b"second-opaque-manifest"
    factory = _Factory(
        scripts=[
            [
                (
                    POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
                    "remote-store",
                    POSTGRESQL_AUTHORITY_CAPABILITY,
                    7,
                ),
                [
                    (
                        "entry-a",
                        "generation-a",
                        "generations/a.native",
                        first_manifest,
                        hashlib.sha256(first_manifest).hexdigest(),
                        1,
                        7,
                    ),
                    (
                        "entry-b",
                        "generation-b",
                        "generations/b.native",
                        second_manifest,
                        hashlib.sha256(second_manifest).hexdigest(),
                        2,
                        7,
                    ),
                ],
            ],
            [
                (
                    POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
                    "remote-store",
                    POSTGRESQL_AUTHORITY_CAPABILITY,
                    8,
                ),
            ],
        ]
    )
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    page = authority.inventory_page(limit=1, work_cap=1024)

    assert page.identity.authority_kind == "postgresql"
    assert page.identity.capability == POSTGRESQL_AUTHORITY_CAPABILITY
    assert page.identity.schema_version == POSTGRESQL_AUTHORITY_SCHEMA_VERSION
    assert tuple(entry.key for entry in page.entries) == ("entry-a",)
    assert page.entries[0].manifest == first_manifest
    assert page.exhausted is False
    assert page.next_cursor is not None

    with pytest.raises(CacheBlobLifecycleConflictError, match="reinspection"):
        authority.inventory_page(cursor=page.next_cursor, limit=1, work_cap=1024)

    page_query, page_params = factory.connections[0].executions[-1]
    assert "order by key, generation" in _query_text(page_query)
    assert page_params == (2,)


@pytest.mark.parametrize("sqlstate", ("40001", "40P01", "55P03", "57014"))
def test_inventory_preserves_retryable_postgresql_progress_causes(sqlstate: str) -> None:
    """Inventory maps declared progress outcomes without skipping a failed row."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    class DriverFailure(Exception):
        pass

    failure = DriverFailure("deterministic inventory failure")
    failure.sqlstate = sqlstate
    authority = PostgresqlLifecycleAuthority(
        _Factory(scripts=[[failure]]), schema="phase5_authority"
    )

    with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
        authority.inventory_page(limit=1, work_cap=1024)

    assert raised.value.__cause__ is failure
    assert raised.value.context["progress_outcome"] in {
        "serialization",
        "deadlock",
        "lock_timeout",
        "statement_timeout",
    }


def test_reconciliation_cursor_does_not_skip_unemitted_debt_rows() -> None:
    """Debt rows fetched beside a full mutation page resume from their old cursor."""
    from cacheness.config import LifecycleLimits
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )
    from cacheness.storage.lifecycle_authority import ReconciliationSnapshot

    mutation = (
        1,
        "operation-1",
        "key-1",
        "generation-1",
        "generations/1.native",
        None,
        None,
        None,
        None,
        b"a",
        "prepared",
    )
    debt = (
        1,
        "debt-operation",
        "generations/debt.native",
        "debt-key",
        "debt-generation",
        "candidate",
        "pending",
    )
    factory = _Factory(scripts=[[[mutation], None, [debt]], [[], None, [debt]]])
    authority = PostgresqlLifecycleAuthority(
        factory,
        schema="phase5_authority",
        lifecycle_limits=LifecycleLimits(operation_page_size=1),
    )
    snapshot = ReconciliationSnapshot(0, 1, 1)

    first_page = authority.page_reconciliation_work(
        snapshot, mutation_cursor=0, debt_cursor=0
    )
    resumed_page = authority.page_reconciliation_work(
        snapshot,
        mutation_cursor=first_page.mutation_cursor,
        debt_cursor=first_page.debt_cursor,
    )

    assert tuple(work.source for work in first_page.works) == ("mutation",)
    assert first_page.debt_cursor == 0
    assert tuple(work.source for work in resumed_page.works) == ("debt",)
    assert resumed_page.debt_cursor == 1


def test_reconciliation_surfaces_an_unsupported_debt_state_before_pending_work() -> None:
    """An invalid durable debt state cannot be skipped by pending-only paging."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )
    from cacheness.storage.lifecycle_authority import ReconciliationSnapshot

    invalid_debt = (
        1,
        "invalid-operation",
        "generations/invalid.native",
        "invalid-key",
        "invalid-generation",
        "candidate",
        "corrupt",
    )
    factory = _Factory(scripts=[[[], invalid_debt, []]])
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    page = authority.page_reconciliation_work(
        ReconciliationSnapshot(0, 0, 1), mutation_cursor=0, debt_cursor=0
    )

    assert len(page.works) == 1
    assert page.works[0].source == "debt"
    assert page.works[0].state == "corrupt"
    assert page.debt_cursor == 1
    statements = [_query_text(query) for query, _ in factory.connections[0].executions]
    assert any("state = 'pending'" in statement for statement in statements)
    assert any("state <> 'pending'" in statement for statement in statements)


def test_complete_authority_protocol_has_no_placeholder_transitions() -> None:
    """The remote authority supplies every engine primitive without a second coordinator."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )
    from cacheness.storage.lifecycle_authority import LifecycleAuthority

    authority = PostgresqlLifecycleAuthority(_Factory(), schema="phase5_authority")

    assert isinstance(authority, LifecycleAuthority)
    for method_name in (
        "list_entries",
        "catalog_page",
        "pending_cleanup_debts",
        "pending_mutations",
        "retire_cleanup_debt",
        "delete_entry",
        "retire_tombstone",
        "begin_clear",
        "page_clear",
        "checkpoint_clear",
        "begin_reconciliation",
        "reconciliation_snapshot",
        "page_reconciliation_work",
        "page_reconciliation",
        "checkpoint_reconciliation",
        "compare_and_mark_projection",
        "projection_backup",
    ):
        assert callable(getattr(authority, method_name))


def test_complete_workflow_calls_stay_at_the_authority_boundary() -> None:
    """Cleanup, clear, and projection actions remain SQL-only bounded primitives."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    debt = CleanupDebt(
        "operation-1",
        "generations/generation-1.native",
        "authority-key",
        "generation-1",
        "candidate",
        1,
    )
    factory = _Factory(
        scripts=[
            [[(1, debt.operation_id, debt.locator, debt.key, debt.generation, debt.role)]],
            [None],
            [None, (7,), None],
            [(7, "", True), ("active", ""), []],
            [("active", True), ("clear-run",)],
            [None, (0,), (0,), (7,), None],
            [(7, 0, 0)],
            [(7, True), (7,)],
        ]
    )
    authority = PostgresqlLifecycleAuthority(factory, schema="phase5_authority")

    assert authority.pending_cleanup_debts() == (debt,)
    authority.retire_cleanup_debt(debt)
    clear = authority.begin_clear()
    assert isinstance(clear, PageToken)
    assert authority.page_clear(clear) == ()
    authority.checkpoint_clear(clear)
    reconciliation = authority.begin_reconciliation()
    assert authority.reconciliation_snapshot(reconciliation).authority_revision == 7
    authority.compare_and_mark_projection(ProjectionRevision(7))

    statements = [
        _query_text(query)
        for connection in factory.connections
        for query, _ in connection.executions
    ]
    assert not any("advisory" in statement or "listen" in statement for statement in statements)
    assert not any("s3" in statement or "boto" in statement for statement in statements)


# =============================================================================
# Explicit offline whole-store publication contracts
# =============================================================================


def _verified_empty_candidate() -> tuple[VerifiedCandidateReceipt, tuple[AuthorityInventoryEntry, ...]]:
    """Build a complete bounded candidate without a payload or service dependency."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        POSTGRESQL_AUTHORITY_CAPABILITY,
        POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
    )

    identity = AuthorityIdentitySnapshot(
        store_id="remote-store",
        revision=7,
        authority_kind="postgresql",
        capability=POSTGRESQL_AUTHORITY_CAPABILITY,
        schema_version=POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
    )
    entries: tuple[AuthorityInventoryEntry, ...] = ()
    return (
        VerifiedCandidateReceipt(
            run_id="run-remote-cutover",
            plan_digest="a" * 64,
            source_identity=identity,
            source_revision=7,
            destination_identity=identity,
            destination_revision=7,
            candidate_digest=candidate_digest(entries),
            entry_count=0,
            byte_count=0,
        ),
        entries,
    )


def test_remote_maintenance_transitions_are_sql_only_and_receipt_bound() -> None:
    """Deterministic PostgreSQL coverage is not a live-service qualification."""
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        POSTGRESQL_AUTHORITY_CAPABILITY,
        POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
        PostgresqlLifecycleAuthority,
    )

    receipt, entries = _verified_empty_candidate()
    factory = _Factory(
        scripts=[[
            (
                POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
                "remote-store",
                POSTGRESQL_AUTHORITY_CAPABILITY,
                7,
            ),
            (7, None, None, None, None, None, "idle", "source", False),
            (7,),
        ]]
    )
    authority = PostgresqlLifecycleAuthority(factory, schema="phase7_authority")

    assert authority.record_verified_candidate(receipt=receipt, entries=entries) == receipt
    assert factory.connections[0].transaction_count == 1
    statements = [_query_text(query) for query, _ in factory.connections[0].executions]
    assert any("migration_state = 'candidate'" in statement for statement in statements)
    assert not any("s3" in statement or "boto" in statement for statement in statements)


def test_activated_offline_state_fences_workers_until_explicit_resolution() -> None:
    """Only maintenance rollback/finalize may resolve the remote offline fence."""
    from cacheness.error_handling import CacheBlobMigrationOfflineDecisionRequiredError
    from cacheness.storage.backends.postgresql_lifecycle_authority import (
        PostgresqlLifecycleAuthority,
    )

    authority = PostgresqlLifecycleAuthority(
        _Factory(scripts=[[(7, "run-remote-cutover", "a" * 64, "b" * 64, 7, 8,
                            AuthorityPublicationState.ACTIVATED_OFFLINE.value,
                            "candidate", True)]]),
        schema="phase7_authority",
    )
    authority.store_identity = "remote-store"

    with pytest.raises(CacheBlobMigrationOfflineDecisionRequiredError):
        authority.require_ordinary_worker_access()
