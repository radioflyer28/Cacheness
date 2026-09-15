"""Phase 8 contracts for lifecycle integrity, recovery, and PostgreSQL boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
import sqlite3
from typing import Any

from obstore.store import MemoryStore
import pytest

from cacheness import CacheConfig
from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobPayloadTamperedError,
    CacheBlobRecoverableCleanupError,
)
from cacheness.storage import BlobStore
from cacheness.storage.backends.postgresql_lifecycle_authority import (
    POSTGRESQL_AUTHORITY_CAPABILITY,
    POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
    PostgresqlLifecycleAuthority,
)
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.lifecycle_authority import (
    EntryExpectation,
    MutationSpec,
    PreparedMutation,
    ReconciliationSnapshot,
    VerificationProof,
)
from cacheness.storage.obstore_generation_io import ObstoreGenerationIO
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


def _query_text(query: object) -> str:
    """Normalize the deterministic DB-API transcript query representation."""

    return str(query).lower()


@dataclass
class _DriverCursor:
    """Minimal DB-API cursor with scripted bounded responses."""

    connection: "_DriverConnection"
    row: tuple[Any, ...] | None = None
    rows: list[tuple[Any, ...]] = field(default_factory=list)

    def execute(self, query: object, params: object = None) -> "_DriverCursor":
        self.connection.executions.append((query, params))
        if "set local" in _query_text(query):
            return self
        self.row = None
        self.rows = []
        response = (
            self.connection.responses.pop(0) if self.connection.responses else None
        )
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            self.row = response
        elif isinstance(response, list):
            self.rows = response
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        """Return one scripted row."""

        return self.row

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Return one scripted row page."""

        return self.rows

    def close(self) -> None:
        """Mirror the small close surface used by the authority."""


@dataclass
class _DriverTransaction:
    """Record commit versus rollback without presenting a live PostgreSQL claim."""

    connection: "_DriverConnection"

    def __enter__(self) -> "_DriverTransaction":
        self.connection.transaction_count += 1
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is None:
            self.connection.commit_count += 1
        else:
            self.connection.rollback_count += 1
        return False


@dataclass
class _DriverConnection:
    """Scripted driver connection whose transaction exits are observable."""

    responses: list[object] = field(default_factory=list)
    executions: list[tuple[object, object]] = field(default_factory=list)
    transaction_count: int = 0
    commit_count: int = 0
    rollback_count: int = 0
    closed: bool = False

    def cursor(self) -> _DriverCursor:
        """Return the recording cursor."""

        return _DriverCursor(self)

    def transaction(self) -> _DriverTransaction:
        """Return one transaction transcript context."""

        return _DriverTransaction(self)

    def close(self) -> None:
        """Record resource ownership release."""

        self.closed = True


@dataclass
class _DriverFactory:
    """Produce a fresh scripted connection for each short authority transaction."""

    scripts: list[list[object]] = field(default_factory=list)
    connections: list[_DriverConnection] = field(default_factory=list)

    def __call__(self) -> _DriverConnection:
        connection = _DriverConnection(
            responses=self.scripts.pop(0) if self.scripts else []
        )
        self.connections.append(connection)
        return connection


class _DriverFailure(Exception):
    """DB-API-like failure with optional SQLSTATE evidence."""

    def __init__(self, message: str, *, sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


OperationalError = type("OperationalError", (Exception,), {})


class _SqlText(str):
    """Minimal composable SQL record used by deterministic DB-API contracts."""

    def format(self, *_identifiers: object) -> "_SqlText":
        """Keep identifier composition observable without evaluating SQL."""

        return self

    def __add__(self, other: object) -> "_SqlText":
        """Preserve the query text required by the transcript assertions."""

        return _SqlText(f"{self}{other}")


class _SqlComposer:
    """Supply the tiny psycopg.sql surface exercised by the no-service fixture."""

    @staticmethod
    def SQL(value: str) -> _SqlText:
        """Create one composable query record."""

        return _SqlText(value)

    @staticmethod
    def Identifier(value: str) -> str:
        """Retain the trusted test identifier without interpolating it."""

        return value


@pytest.fixture(autouse=True)
def _deterministic_postgresql_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Phase 8 DB-API contracts independent of a live optional driver."""

    import cacheness.storage.backends.postgresql_lifecycle_authority as module

    monkeypatch.setattr(module, "sql", _SqlComposer)


def _postgresql_authority(
    *scripts: list[object], limits: LifecycleLimits | None = None
) -> tuple[PostgresqlLifecycleAuthority, _DriverFactory]:
    """Build a non-live authority from explicit DB-API transcripts."""

    factory = _DriverFactory(scripts=list(scripts))
    authority = PostgresqlLifecycleAuthority(
        factory,
        schema="phase8_authority",
        lifecycle_limits=limits,
    )
    return authority, factory


def _mutation_spec(
    operation_id: str = "phase8-operation",
    *,
    locator: str = "generations/phase8/one.native",
) -> MutationSpec:
    """Create one exact immutable operation descriptor for replay tests."""

    return MutationSpec.create(
        operation_id=operation_id,
        key="phase8-key",
        generation="phase8-generation",
        candidate_locator=locator,
        expected=EntryExpectation.absent(),
        manifest=b"phase8-canonical-manifest",
    )


def _existing_mutation_values(spec: MutationSpec) -> tuple[object, ...]:
    """Render the immutable persistence tuple checked by operation replay."""

    return (
        spec.key,
        spec.generation,
        spec.candidate_locator,
        spec.expected.lineage,
        spec.expected.revision,
        spec.expected.generation,
        spec.expected.manifest_digest,
        spec.manifest,
    )


def _filesystem_store(
    root: Path, *, limits: LifecycleLimits | None = None
) -> BlobStore:
    """Build the qualified local topology for deterministic lifecycle contracts."""

    config = None if limits is None else CacheConfig(lifecycle_limits=limits)
    return BlobStore(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        ),
        cache_dir=root,
        config=config,
    )


@dataclass(frozen=True)
class _ByteRecord:
    """One handler-owned native payload for exact obstore publication tests."""

    value: bytes


class _ByteHandler:
    """Minimal path-based handler that remains private to the participant stage."""

    data_type = "phase8_bytes"
    payload_format = "bytes"
    payload_format_version = 1

    def put(
        self, data: _ByteRecord, file_path: Path, _config: object
    ) -> dict[str, object]:
        """Write one private byte artifact."""

        payload = file_path.with_suffix(".bytes")
        payload.write_bytes(data.value)
        return {
            "actual_path": str(payload),
            "file_size": payload.stat().st_size,
            "metadata": {},
            "payload_format": self.payload_format,
            "payload_format_version": self.payload_format_version,
        }

    def get(self, file_path: Path, _metadata: dict[str, object]) -> _ByteRecord:
        """Read an already-verified private snapshot."""

        return _ByteRecord(file_path.read_bytes())


class _RecordingMemoryStore:
    """Forward exact object calls while rejecting inventory-based settlement."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.calls: list[tuple[str, str]] = []

    def put(self, locator: str, source: object, **kwargs: object) -> object:
        """Forward one immutable create."""

        self.calls.append(("put", locator))
        return self.store.put(locator, source, **kwargs)

    def get(self, locator: str) -> object:
        """Forward one exact object read."""

        self.calls.append(("get", locator))
        return self.store.get(locator)

    def head(self, locator: str) -> object:
        """Forward one exact object observation."""

        self.calls.append(("head", locator))
        return self.store.head(locator)

    def delete(self, locators: list[str]) -> object:
        """Forward exact deletion."""

        return self.store.delete(locators)

    def list(self, *args: object, **kwargs: object) -> object:
        """Forbid listing as a source of publication truth."""

        del args, kwargs
        raise AssertionError("ambiguous publication must not use object inventory")


class _AcceptedThenLostResponseStore(_RecordingMemoryStore):
    """Persist an immutable create while dropping its response."""

    def put(self, locator: str, source: object, **kwargs: object) -> object:
        super().put(locator, source, **kwargs)
        raise OSError("accepted create response was lost")


class _MismatchedThenLostResponseStore(_RecordingMemoryStore):
    """Persist conflicting bytes before returning an ambiguous transport error."""

    def put(self, locator: str, source: object, **kwargs: object) -> object:
        self.calls.append(("put", locator))
        payload = source.read()
        self.store.put(locator, b"x" * len(payload), **kwargs)
        raise OSError("accepted create response was lost")


def _obstore_participant(root: Path, store: object) -> ObstoreGenerationIO:
    """Build one memory participant with handler paths kept beneath the test root."""

    handler_root = root / "handler-stage"
    handler_root.mkdir()
    return ObstoreGenerationIO(
        store,
        GuardedHandlerIO(handler_root),
        qualification_identity="memory",
    )


@pytest.mark.parametrize(
    ("sqlstate", "outcome"),
    (
        ("40001", "serialization"),
        ("40P01", "deadlock"),
        ("55P03", "lock_timeout"),
        ("57014", "statement_timeout"),
    ),
)
def test_postgresql_error_classification_preserves_typed_progress_outcomes(
    sqlstate: str, outcome: str
) -> None:
    """Declared PostgreSQL SQLSTATEs remain retryable progress, not corruption."""

    failure = _DriverFailure("retryable driver failure", sqlstate=sqlstate)
    authority, _ = _postgresql_authority([failure])

    with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
        authority.inventory_page(limit=1, work_cap=1024)

    assert raised.value.context["progress_outcome"] == outcome
    assert raised.value.context["sqlstate"] == sqlstate
    assert raised.value.__cause__ is failure


def test_postgresql_error_classification_handles_driver_classes_and_redacts_unknowns() -> (
    None
):
    """Known driver class names stay retryable while unknown details remain chained only."""

    connection_failure = OperationalError("transient connection loss")
    retryable_authority, _ = _postgresql_authority([connection_failure])

    with pytest.raises(CacheBlobLifecycleTimeoutError) as retryable:
        retryable_authority.inventory_page(limit=1, work_cap=1024)

    assert retryable.value.context["progress_outcome"] == "connection_timeout"
    assert retryable.value.__cause__ is connection_failure

    secret = "postgresql://phase8:secret@example.invalid/cacheness"
    unknown = _DriverFailure(secret)
    unknown_authority, _ = _postgresql_authority([unknown])

    with pytest.raises(CacheBlobBackendError) as captured:
        unknown_authority.inventory_page(limit=1, work_cap=1024)

    assert captured.value.__cause__ is unknown
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value.context)


def test_postgresql_replay_accepts_only_identical_operation_and_proof() -> None:
    """Replays keep one immutable descriptor and reject changed locator or proof facts."""

    spec = _mutation_spec()
    authority, factory = _postgresql_authority(
        [None, None, None],
        [_existing_mutation_values(spec)],
    )

    prepared = authority.prepare_mutation(spec)
    assert authority.prepare_mutation(spec) == prepared
    mutation_inserts = [
        _query_text(query)
        for connection in factory.connections
        for query, _ in connection.executions
        if "insert into" in _query_text(query)
    ]
    assert len(mutation_inserts) == 1

    changed = _mutation_spec(locator="generations/phase8/changed.native")
    changed_authority, changed_factory = _postgresql_authority(
        [_existing_mutation_values(spec)]
    )
    with pytest.raises(CacheBlobLifecycleConflictError, match="not reusable"):
        changed_authority.prepare_mutation(changed)
    assert not any(
        "insert into" in _query_text(query)
        for query, _ in changed_factory.connections[0].executions
    )

    proof = VerificationProof(
        "a" * 64,
        len(spec.manifest),
        spec.manifest,
        b"phase8-transport-evidence",
    )
    for changed_proof in (
        replace(proof, digest="b" * 64),
        replace(proof, byte_size=proof.byte_size + 1),
        replace(proof, transport_evidence=b"changed-transport-evidence"),
        replace(proof, manifest=b"changed-proof-bytes"),
    ):
        proof_authority, proof_factory = _postgresql_authority(
            [(spec.operation_id,)],
            [None],
        )
        proof_authority.record_verification(
            PreparedMutation(spec.operation_id, spec), proof
        )
        with pytest.raises(CacheBlobLifecycleConflictError):
            proof_authority.record_verification(
                PreparedMutation(spec.operation_id, spec), changed_proof
            )
        assert not any(
            "insert into" in _query_text(query)
            for connection in proof_factory.connections
            for query, _ in connection.executions
        )


def _inventory_row(key: str, generation: str, manifest: bytes) -> tuple[object, ...]:
    """Build one digest-bound inventory row for cursor tests."""

    return (
        key,
        generation,
        f"generations/phase8/{generation}.native",
        manifest,
        hashlib.sha256(manifest).hexdigest(),
        1,
        7,
        None,
    )


def test_postgresql_pagination_bounds_work_and_keeps_unemitted_rows_reachable() -> None:
    """Cursor pages retain the stable high-water revision and never skip byte-capped rows."""

    first = _inventory_row("a", "one", b"aaaaaa")
    second = _inventory_row("b", "two", b"bbbbbb")
    identity = (
        POSTGRESQL_AUTHORITY_SCHEMA_VERSION,
        "phase8-store",
        POSTGRESQL_AUTHORITY_CAPABILITY,
        7,
    )
    stale_identity = identity[:-1] + (8,)
    authority, _ = _postgresql_authority(
        [identity, [first, second]],
        [identity, [second]],
        [stale_identity],
    )

    first_page = authority.inventory_page(limit=2, work_cap=len(b"aaaaaa"))
    assert tuple(entry.key for entry in first_page.entries) == ("a",)
    assert first_page.next_cursor is not None
    assert first_page.next_cursor.revision == 7

    second_page = authority.inventory_page(
        cursor=first_page.next_cursor,
        limit=2,
        work_cap=len(b"bbbbbb"),
    )
    assert tuple(entry.key for entry in second_page.entries) == ("b",)
    assert second_page.exhausted is True

    with pytest.raises(CacheBlobLifecycleConflictError, match="reinspection"):
        authority.inventory_page(cursor=first_page.next_cursor, limit=1, work_cap=1024)


def test_postgresql_pagination_reconciliation_cursor_reaches_each_emitted_row() -> None:
    """Reconciliation work caps advance only after the matching durable row is emitted."""

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

    first, second = mutation_row(1), mutation_row(2)
    authority, _ = _postgresql_authority(
        [[first, second], []],
        [[second], []],
        limits=LifecycleLimits(operation_page_size=2, max_operation_record_bytes=11),
    )
    snapshot = ReconciliationSnapshot(0, 2, 0)
    page_one = authority.page_reconciliation_work(
        snapshot, mutation_cursor=0, debt_cursor=0
    )
    page_two = authority.page_reconciliation_work(
        snapshot, mutation_cursor=page_one.mutation_cursor, debt_cursor=0
    )

    assert tuple(work.row_id for work in page_one.works) == (1,)
    assert tuple(work.row_id for work in page_two.works) == (2,)
    assert (page_one.mutation_cursor, page_two.mutation_cursor) == (1, 2)


@pytest.mark.parametrize(
    "stage",
    (
        "schema_initialize",
        "prepare_mutation",
        "record_verification",
        "promote_mutation",
        "page_clear",
        "page_reconciliation",
    ),
)
def test_postgresql_transaction_rollback_preserves_no_partial_commit(
    stage: str,
) -> None:
    """Each Phase 8 transaction boundary rolls back faulted authority work."""

    authority, factory = _postgresql_authority([])
    secret = "postgresql://phase8:rollback-secret@example.invalid/cacheness"
    failure = _DriverFailure(secret)

    def fail_after_pending_write(cursor: _DriverCursor) -> None:
        cursor.execute("INSERT phase8_pending_transition")
        raise failure

    with pytest.raises(CacheBlobBackendError) as captured:
        authority._transaction(stage, fail_after_pending_write)

    connection = factory.connections[0]
    assert captured.value.__cause__ is failure
    assert secret not in str(captured.value)
    assert connection.transaction_count == 1
    assert connection.rollback_count == 1
    assert connection.commit_count == 0
    assert any(
        "insert phase8_pending_transition" in _query_text(query)
        for query, _ in connection.executions
    )


def test_lifecycle_integrity_before_handler_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical SHA-256 and size reject corrupted bytes before handler deserialization."""

    import cacheness.storage.lifecycle as lifecycle_module

    store = _filesystem_store(tmp_path / "integrity-before-handler")
    try:
        key = store.put({"phase": 8}, key="integrity-key")
        entry = store.lifecycle_authority.read_entry(key)
        assert entry is not None
        manifest = store._authenticated_authority_manifest(entry.manifest)
        handler = store.handlers.get_handler_by_type(manifest.handler_type)
        events: list[str] = []

        def bad_digest(*_args: object, **_kwargs: object) -> tuple[str, int]:
            events.append("digest")
            return "0" * 64, 0

        def forbidden_handler(*_args: object, **_kwargs: object) -> object:
            events.append("handler")
            raise AssertionError("corrupt bytes reached deserialization")

        monkeypatch.setattr(lifecycle_module, "sha256_and_size", bad_digest)
        monkeypatch.setattr(handler, "get", forbidden_handler)

        with pytest.raises(CacheBlobPayloadTamperedError):
            store.get(key)

        assert events == ["digest"]
    finally:
        store.close()


@pytest.mark.parametrize(
    "wrapper_type, expected_error",
    (
        (_AcceptedThenLostResponseStore, None),
        (_MismatchedThenLostResponseStore, CacheBlobLifecycleConflictError),
    ),
)
def test_lifecycle_ambiguous_publication_settles_only_exact_locator_identity(
    tmp_path: Path,
    wrapper_type: type[_RecordingMemoryStore],
    expected_error: type[BaseException] | None,
) -> None:
    """A lost create response uses exact head/get evidence and rejects mismatched bytes."""

    wrapped = wrapper_type(MemoryStore())
    participant = _obstore_participant(tmp_path, wrapped)
    locator = Path("generations") / "phase8" / "ambiguous.bytes"
    try:
        with participant.stage(
            _ByteHandler(), _ByteRecord(b"phase8"), config=None
        ) as staged:
            if expected_error is None:
                published = participant.publish_generation(staged, locator)
                with participant.open_snapshot(
                    locator, dict(published["metadata"])
                ) as snapshot:
                    assert snapshot.path.read_bytes() == b"phase8"
            else:
                with pytest.raises(expected_error):
                    participant.publish_generation(staged, locator)
        assert ("head", locator.as_posix()) in wrapped.calls
        assert ("get", locator.as_posix()) in wrapped.calls
    finally:
        participant.close()


def test_lifecycle_cleanup_debt_reconciliation_retires_only_exact_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-promotion cleanup debt retains exact identity until deterministic recovery."""

    store = _filesystem_store(tmp_path / "cleanup-debt")
    try:
        key = store.put({"generation": "old"}, key="cleanup-key")
        previous = store.lifecycle_authority.read_entry(key)
        assert previous is not None
        old_payload = store.cache_dir / previous.locator
        original_delete = store._delete_or_prove_absent
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            lambda _locator: (_ for _ in ()).throw(OSError("defer exact cleanup")),
        )

        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.put({"generation": "winner"}, key=key)

        debts = store.lifecycle_authority.pending_cleanup_debts(key=key)
        assert len(debts) == 1
        assert debts[0].locator == previous.locator
        assert old_payload.exists()
        assert store.get(key) == {"generation": "winner"}

        monkeypatch.setattr(store, "_delete_or_prove_absent", original_delete)
        report = store.reconcile(apply=True)
        assert report.applied is True
        assert store.lifecycle_authority.pending_cleanup_debts(key=key) == ()
        assert not old_payload.exists()
        assert store.get(key) == {"generation": "winner"}
    finally:
        store.close()


def test_lifecycle_bounded_clear_preserves_changed_generations(tmp_path: Path) -> None:
    """A bounded clear snapshot conflicts safely with a post-snapshot replacement."""

    limits = LifecycleLimits(max_reconcile_actions=1, operation_page_size=1)
    store = _filesystem_store(tmp_path / "bounded-clear", limits=limits)
    try:
        store.put({"generation": "old"}, key="existing")
        store.put({"generation": "other"}, key="other")
        replaced = False

        def replace_after_snapshot(boundary: str) -> None:
            nonlocal replaced
            if boundary == "clear.snapshot_committed" and not replaced:
                replaced = True
                store.put({"generation": "new"}, key="existing")

        store.lifecycle.test_hook = replace_after_snapshot
        token = store.lifecycle.begin_clear()
        assert store.lifecycle.complete_clear(token) == 1
        assert store.lifecycle.complete_clear(token) == 0
        assert store.get("existing") == {"generation": "new"}
        assert store.get("other") is None
    finally:
        store.close()


def test_sqlite_lifecycle_rejects_invalid_configuration_objects_without_materializing(
    tmp_path: Path,
) -> None:
    """Invalid authority configuration is rejected before a root is created."""

    invalid_limits_root = tmp_path / "invalid-limits"
    with pytest.raises(
        TypeError, match="lifecycle_limits must be a LifecycleLimits instance"
    ):
        SqliteLifecycleAuthority.for_root(
            invalid_limits_root, lifecycle_limits=object()
        )
    assert not invalid_limits_root.exists()

    invalid_topology_root = tmp_path / "invalid-topology"
    with pytest.raises(
        TypeError,
        match="lifecycle_topology must be a LifecycleAuthorityTopology instance",
    ):
        SqliteLifecycleAuthority.for_root(
            invalid_topology_root, lifecycle_topology=object()
        )
    assert not invalid_topology_root.exists()


def test_sqlite_lifecycle_rejects_invalid_deadline_and_busy_budget_inputs(
    tmp_path: Path,
) -> None:
    """Deadline and PRAGMA input validation leaves SQLite and roots unchanged."""

    root = tmp_path / "invalid-inputs"
    authority = SqliteLifecycleAuthority.for_root(root)
    connection = sqlite3.connect(":memory:")
    try:
        for deadline in (True, "not-a-monotonic-timestamp"):
            with pytest.raises(
                TypeError, match="deadline must be a monotonic timestamp"
            ):
                authority._deadline(deadline)
        assert not root.exists()

        before = connection.execute("PRAGMA busy_timeout").fetchone()
        assert before is not None
        for milliseconds in (-1, True, 1.5):
            with pytest.raises(
                ValueError,
                match="SQLite busy timeout must be a non-negative integer",
            ):
                authority._set_busy_timeout(connection, milliseconds)
            assert connection.execute("PRAGMA busy_timeout").fetchone() == before
    finally:
        connection.close()
        authority.close()


def test_sqlite_lifecycle_timeout_and_sqlite_failures_preserve_typed_context(
    tmp_path: Path,
) -> None:
    """Expired budgets and real SQLite errors retain their public failure contracts."""

    authority = SqliteLifecycleAuthority.for_root(
        tmp_path / "deadline-errors",
        lifecycle_limits=LifecycleLimits(authority_busy_timeout_seconds=2.0),
    )
    authority._monotonic_clock = lambda: 10.0
    try:
        with pytest.raises(CacheBlobLifecycleTimeoutError) as timeout:
            authority._remaining_for_stage(
                10.0,
                stage="phase8_expired_budget",
                started_at=8.0,
            )

        assert str(timeout.value) == "Lifecycle authority busy deadline expired"
        assert timeout.value.context == {
            "operation": "lifecycle_authority",
            "stage": "phase8_expired_budget",
            "elapsed_seconds": 2.0,
            "remaining_seconds": 0.0,
            "authority_busy_timeout_seconds": 2.0,
            "authority_path": str(authority.path),
            "retryable": True,
            "reason": "blob_lifecycle_timeout",
        }

        closed_connection = sqlite3.connect(":memory:")
        closed_connection.close()
        with pytest.raises(CacheBlobBackendError) as backend_error:
            authority._apply_stage_busy_timeout(
                closed_connection,
                deadline=12.0,
                started_at=10.0,
                stage="phase8_closed_connection",
            )

        assert str(backend_error.value) == "Lifecycle authority SQLite operation failed"
        assert backend_error.value.context == {
            "operation": "lifecycle_authority",
            "reason": "blob_backend_failure",
        }
        assert isinstance(backend_error.value.__cause__, sqlite3.ProgrammingError)
    finally:
        authority.close()
