"""Direct PostgreSQL lifecycle authority primitives for the remote topology.

This adapter owns only the transactional visibility half of a BlobStore
lifecycle.  It never stages, publishes, verifies, lists, or deletes external
payload objects: :class:`AuthorityLifecycleEngine` keeps that ordering above
this narrow database boundary.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
import re
from typing import Any, Callable, Iterator, TypeVar
import uuid

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobMigrationRequiredError,
    CacheBlobStoreClosedError,
)
from cacheness.storage.lifecycle_authority import (
    AuthorityCapabilities,
    AuthorityStateSnapshot,
    CleanupDebt,
    EntryExpectation,
    EntrySnapshot,
    MutationSpec,
    PreparedMutation,
    PromotionResult,
    VerificationProof,
)

try:  # Keep the package importable without the optional PostgreSQL extra.
    from psycopg import errors, sql
except ImportError:  # pragma: no cover - exercised by optional-dependency users.
    errors = None
    sql = None


POSTGRESQL_AUTHORITY_SCHEMA_VERSION = 1
"""Current PostgreSQL authority layout version; Phase 7 migration input."""

POSTGRESQL_AUTHORITY_CAPABILITY = "postgresql-lifecycle-authority-v1"
"""Persisted authority capability marker, independent of payload formats."""

SCHEMA_VERSION = POSTGRESQL_AUTHORITY_SCHEMA_VERSION
"""Short diagnostic alias for the current persisted PostgreSQL schema only."""

_SCHEMA_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}\Z")
_MAX_STORE_IDENTITY_BYTES = 64
_RETRYABLE_SQLSTATES = frozenset({"40001", "40P01", "55P03", "57014"})
_REQUIRED_TABLES = frozenset(
    {
        "authority_meta",
        "entry_lineage",
        "entries",
        "mutations",
        "cleanup_debt",
        "clear_runs",
        "clear_targets",
        "reconciliation_runs",
        "reconciliation_actions",
    }
)
_REQUIRED_CONSTRAINTS = frozenset(
    {
        "authority_meta_singleton_check",
        "mutations_operation_id_key",
        "cleanup_debt_operation_locator_role_key",
        "clear_targets_run_id_key_key",
        "reconciliation_actions_run_id_action_id_key",
    }
)
_T = TypeVar("_T")


def _bounded_identity(value: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_STORE_IDENTITY_BYTES:
        raise ValueError("store_identity must be a non-empty bounded string")
    return value


@dataclass(frozen=True)
class _MutationRow:
    """Decoded mutation row retained only inside one authority transaction."""

    spec: MutationSpec
    verified_digest: str | None
    verified_size: int | None
    state: str


class PostgresqlLifecycleAuthority:
    """One direct-psycopg authority with one fresh connection lease per operation.

    ``connection_factory`` is supplied by the application or a caller-owned
    pool.  It must return one psycopg connection (or a pool lease context
    manager) on each invocation.  The authority stores neither a connection
    nor a DSN, so forked workers cannot accidentally share a socket.
    """

    capabilities = AuthorityCapabilities(durable=True, multiprocess=True)
    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "transaction_scope": "authority",
        "exact_cas": True,
        "portable_query": True,
        "canonical_scan": True,
        "index_acceleration": True,
    }

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        *,
        schema: str = "cacheness_authority",
        store_identity: str | None = None,
        statement_timeout_ms: int = 5_000,
        lock_timeout_ms: int = 1_000,
    ) -> None:
        if sql is None:
            raise ImportError(
                "PostgreSQL lifecycle authority requires psycopg. "
                "Install with: uv sync --extra postgresql"
            )
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        if not isinstance(schema, str) or not _SCHEMA_NAME.fullmatch(schema):
            raise ValueError("schema must be a bounded PostgreSQL identifier")
        for field_name, value in (
            ("statement_timeout_ms", statement_timeout_ms),
            ("lock_timeout_ms", lock_timeout_ms),
        ):
            if type(value) is not int or value <= 0 or value > 60_000:
                raise ValueError(f"{field_name} must be an integer between 1 and 60000")
        self._connection_factory = connection_factory
        self.schema = schema
        self._expected_store_identity = (
            None if store_identity is None else _bounded_identity(store_identity)
        )
        self.store_identity: str | None = None
        self.statement_timeout_ms = statement_timeout_ms
        self.lock_timeout_ms = lock_timeout_ms
        self._owner_pid = os.getpid()
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise CacheBlobStoreClosedError("PostgreSQL lifecycle authority is closed")
        if self._owner_pid != os.getpid():
            raise CacheBlobBackendError(
                "PostgreSQL lifecycle authority cannot be reused after fork",
                context={"operation": "postgresql_lifecycle_authority"},
            )

    def _table(self, name: str) -> Any:
        """Compose the only dynamic SQL values as psycopg identifiers."""
        return sql.SQL("{}.{}").format(sql.Identifier(self.schema), sql.Identifier(name))

    @contextmanager
    def _lease(self) -> Iterator[Any]:
        """Lease, then release, exactly one caller-owned connection resource."""
        self._require_open()
        resource = self._connection_factory()
        connection: Any | None = None
        entered = False
        try:
            if hasattr(resource, "__enter__") and hasattr(resource, "__exit__"):
                connection = resource.__enter__()
                entered = True
            else:
                connection = resource
            if connection is None or not hasattr(connection, "cursor"):
                raise TypeError("connection_factory must return a psycopg connection or lease")
            yield connection
        finally:
            if entered:
                resource.__exit__(None, None, None)
            elif connection is not None:
                close = getattr(connection, "close", None)
                if callable(close):
                    close()

    def _configure_transaction(self, cursor: Any) -> None:
        """Bound one transaction locally; no process or cluster lock is involved."""
        cursor.execute("SET LOCAL statement_timeout = %s", (f"{self.statement_timeout_ms}ms",))
        cursor.execute("SET LOCAL lock_timeout = %s", (f"{self.lock_timeout_ms}ms",))

    @staticmethod
    def _sqlstate(error: BaseException) -> str | None:
        value = getattr(error, "sqlstate", None)
        return value if isinstance(value, str) else None

    def _raise_driver_error(self, error: BaseException, *, stage: str) -> None:
        """Map only bounded, non-secret driver context into public exceptions."""
        sqlstate = self._sqlstate(error)
        context = {"operation": "postgresql_lifecycle_authority", "stage": stage}
        if sqlstate in _RETRYABLE_SQLSTATES:
            context["sqlstate"] = sqlstate
            raise CacheBlobLifecycleTimeoutError(
                "PostgreSQL lifecycle authority made no bounded progress", context=context
            ) from error
        if errors is not None and isinstance(
            error,
            (
                errors.SerializationFailure,
                errors.DeadlockDetected,
                errors.LockNotAvailable,
                errors.QueryCanceled,
            ),
        ):
            raise CacheBlobLifecycleTimeoutError(
                "PostgreSQL lifecycle authority made no bounded progress", context=context
            ) from error
        raise CacheBlobBackendError(
            "PostgreSQL lifecycle authority operation failed", context=context
        ) from error

    def _transaction(self, stage: str, action: Callable[[Any], _T]) -> _T:
        """Run one semantic transition in one short PostgreSQL transaction."""
        try:
            with self._lease() as connection:
                with connection.transaction():
                    cursor = connection.cursor()
                    try:
                        self._configure_transaction(cursor)
                        return action(cursor)
                    finally:
                        close = getattr(cursor, "close", None)
                        if callable(close):
                            close()
        except (
            CacheBlobBackendError,
            CacheBlobLifecycleConflictError,
            CacheBlobLifecycleTimeoutError,
            CacheBlobMigrationRequiredError,
            CacheBlobStoreClosedError,
            TypeError,
            ValueError,
        ):
            raise
        except BaseException as error:
            self._raise_driver_error(error, stage=stage)
            raise AssertionError("unreachable")

    def _read_only(self, stage: str, action: Callable[[Any], _T]) -> _T:
        """Perform validation/reads without DDL or mutation statements."""
        return self._transaction(stage, action)

    def _metadata(self, cursor: Any) -> tuple[int, str, str] | None:
        cursor.execute(
            sql.SQL(
                "SELECT schema_version, store_identity, capability "
                "FROM {} WHERE singleton = TRUE"
            ).format(self._table("authority_meta"))
        )
        return cursor.fetchone()

    def _validate_schema(self, cursor: Any) -> None:
        """Validate persisted version, identity, capability, tables, and constraints."""
        try:
            metadata = self._metadata(cursor)
        except BaseException as error:
            if errors is not None and isinstance(error, errors.UndefinedTable):
                raise CacheBlobMigrationRequiredError(
                    "PostgreSQL lifecycle authority layout requires explicit migration or rebuild"
                ) from error
            raise
        if metadata is None or len(metadata) != 3:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority store identity is incompatible"
            )
        version, identity, capability = metadata
        if type(version) is not int or version != POSTGRESQL_AUTHORITY_SCHEMA_VERSION:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority schema version is incompatible"
            )
        try:
            identity = _bounded_identity(identity)
        except ValueError as error:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority store identity is incompatible"
            ) from error
        if self._expected_store_identity is not None and identity != self._expected_store_identity:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority store identity is incompatible"
            )
        if capability != POSTGRESQL_AUTHORITY_CAPABILITY:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority capability is incompatible"
            )
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = ANY(%s)",
            (self.schema, list(_REQUIRED_TABLES)),
        )
        table_names = {row[0] for row in cursor.fetchall()}
        if table_names != _REQUIRED_TABLES:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority table layout is incompatible"
            )
        cursor.execute(
            "SELECT constraint_name FROM information_schema.table_constraints "
            "WHERE table_schema = %s AND constraint_name = ANY(%s)",
            (self.schema, list(_REQUIRED_CONSTRAINTS)),
        )
        constraint_names = {row[0] for row in cursor.fetchall()}
        if constraint_names != _REQUIRED_CONSTRAINTS:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority constraint layout is incompatible"
            )
        self.store_identity = identity

    def open(self) -> "PostgresqlLifecycleAuthority":
        """Validate an existing authority without creating, upgrading, or repairing it."""
        self._read_only("schema_validate", self._validate_schema)
        return self

    def initialize(self) -> None:
        """Create only the exact current schema at an explicit stopped-worker boundary."""
        for attempt in range(2):
            try:
                self._transaction("schema_initialize", self._initialize_once)
                return
            except CacheBlobLifecycleTimeoutError:
                if attempt == 1:
                    raise
            except CacheBlobBackendError as error:
                if self._sqlstate(error.__cause__ or error) != "23505" or attempt == 1:
                    raise
        raise AssertionError("bounded PostgreSQL initialization loop exhausted")

    def _initialize_once(self, cursor: Any) -> None:
        """Emit current-version DDL only in the caller-selected initialization action."""
        schema = sql.Identifier(self.schema)
        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "singleton BOOLEAN PRIMARY KEY DEFAULT TRUE, "
                "schema_version INTEGER NOT NULL, store_identity TEXT NOT NULL, "
                "capability TEXT NOT NULL, authority_revision BIGINT NOT NULL, "
                "projection_dirty BOOLEAN NOT NULL, "
                "CONSTRAINT authority_meta_singleton_check CHECK (singleton = TRUE))"
            ).format(self._table("authority_meta"))
        )
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (key TEXT PRIMARY KEY, lineage BIGINT NOT NULL)"
            ).format(self._table("entry_lineage"))
        )
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "key TEXT PRIMARY KEY, generation TEXT NOT NULL, locator TEXT NOT NULL, "
                "manifest BYTEA NOT NULL, manifest_digest TEXT NOT NULL, lineage BIGINT NOT NULL, "
                "revision BIGINT NOT NULL)"
            ).format(self._table("entries"))
        )
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "operation_id TEXT PRIMARY KEY, key TEXT NOT NULL, generation TEXT NOT NULL, "
                "locator TEXT NOT NULL, expected_lineage BIGINT, expected_revision BIGINT, "
                "expected_generation TEXT, expected_manifest_digest TEXT, manifest BYTEA NOT NULL, "
                "verified_digest TEXT, verified_size BIGINT, state TEXT NOT NULL, "
                "CONSTRAINT mutations_operation_id_key UNIQUE (operation_id))"
            ).format(self._table("mutations"))
        )
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "debt_id BIGSERIAL PRIMARY KEY, operation_id TEXT NOT NULL, key TEXT NOT NULL, "
                "generation TEXT NOT NULL, locator TEXT NOT NULL, role TEXT NOT NULL, state TEXT NOT NULL, "
                "CONSTRAINT cleanup_debt_operation_locator_role_key "
                "UNIQUE (operation_id, locator, role))"
            ).format(self._table("cleanup_debt"))
        )
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "run_id TEXT PRIMARY KEY, state TEXT NOT NULL, revision BIGINT NOT NULL, "
                "last_key TEXT NOT NULL DEFAULT '')"
            ).format(self._table("clear_runs"))
        )
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "run_id TEXT NOT NULL, key TEXT NOT NULL, lineage BIGINT NOT NULL, "
                "entry_revision BIGINT NOT NULL, generation TEXT NOT NULL, locator TEXT NOT NULL, "
                "manifest BYTEA NOT NULL, manifest_digest TEXT NOT NULL, state TEXT NOT NULL, "
                "CONSTRAINT clear_targets_run_id_key_key PRIMARY KEY (run_id, key))"
            ).format(self._table("clear_targets"))
        )
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "run_id TEXT PRIMARY KEY, state TEXT NOT NULL, mutation_high_water BIGINT NOT NULL, "
                "debt_high_water BIGINT NOT NULL, authority_revision BIGINT NOT NULL, "
                "mutation_cursor BIGINT NOT NULL DEFAULT 0, debt_cursor BIGINT NOT NULL DEFAULT 0)"
            ).format(self._table("reconciliation_runs"))
        )
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "run_id TEXT NOT NULL, action_id BIGINT NOT NULL, state TEXT NOT NULL, "
                "CONSTRAINT reconciliation_actions_run_id_action_id_key PRIMARY KEY (run_id, action_id))"
            ).format(self._table("reconciliation_actions"))
        )
        identity = self._expected_store_identity or uuid.uuid4().hex
        cursor.execute(
            sql.SQL(
                "INSERT INTO {} (singleton, schema_version, store_identity, capability, "
                "authority_revision, projection_dirty) VALUES (TRUE, %s, %s, %s, 0, FALSE) "
                "ON CONFLICT (singleton) DO NOTHING"
            ).format(self._table("authority_meta")),
            (POSTGRESQL_AUTHORITY_SCHEMA_VERSION, identity, POSTGRESQL_AUTHORITY_CAPABILITY),
        )
        # A created layout is known to have the exact table/constraint definitions.
        # Existing layouts are checked separately by ``open()`` without DDL.
        self.store_identity = identity

    def preflight_mutation(self) -> None:
        """Require explicit successful initialization before mutation work starts."""
        self.open()

    @staticmethod
    def _matches(expected: EntryExpectation, observed: EntryExpectation) -> bool:
        return expected == observed

    def _expectation(self, cursor: Any, key: str, *, lock: bool = False) -> EntryExpectation:
        suffix = " FOR UPDATE OF l, e" if lock else ""
        cursor.execute(
            sql.SQL(
                "SELECT l.lineage, e.revision, e.generation, e.manifest_digest "
                "FROM {} AS l LEFT JOIN {} AS e ON e.key = l.key "
                "WHERE l.key = %s"
            ).format(self._table("entry_lineage"), self._table("entries"))
            + sql.SQL(suffix),
            (key,),
        )
        row = cursor.fetchone()
        if row is None:
            return EntryExpectation.absent()
        lineage, revision, generation, manifest_digest = row
        return EntryExpectation(lineage, revision, generation, manifest_digest)

    def read_entry(self, key: str) -> EntrySnapshot | None:
        def read(cursor: Any) -> EntrySnapshot | None:
            cursor.execute(
                sql.SQL(
                    "SELECT generation, locator, manifest, manifest_digest, lineage, revision "
                    "FROM {} WHERE key = %s"
                ).format(self._table("entries")),
                (key,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            generation, locator, manifest, digest, lineage, revision = row
            manifest = bytes(manifest)
            if hashlib.sha256(manifest).hexdigest() != digest:
                raise CacheBlobBackendError(
                    "PostgreSQL lifecycle authority entry row is malformed",
                    context={"operation": "postgresql_lifecycle_authority_read"},
                )
            return EntrySnapshot(
                key, generation, locator, manifest,
                EntryExpectation(lineage, revision, generation, digest),
            )

        return self._read_only("read_entry", read)

    def read_expectation(self, key: str) -> EntryExpectation:
        return self._read_only("read_expectation", lambda cursor: self._expectation(cursor, key))

    def snapshot_state(self) -> AuthorityStateSnapshot:
        def snapshot(cursor: Any) -> AuthorityStateSnapshot:
            cursor.execute(
                sql.SQL("SELECT authority_revision, projection_dirty FROM {} WHERE singleton = TRUE").format(
                    self._table("authority_meta")
                )
            )
            meta = cursor.fetchone()
            if meta is None:
                raise CacheBlobMigrationRequiredError("PostgreSQL lifecycle authority is absent")
            cursor.execute(sql.SQL("SELECT operation_id, state FROM {} ORDER BY operation_id").format(self._table("mutations")))
            states = tuple((str(row[0]), str(row[1])) for row in cursor.fetchall())
            return AuthorityStateSnapshot(int(meta[0]), bool(meta[1]), states, ())

        return self._read_only("snapshot_state", snapshot)

    def prepare_mutation(self, spec: MutationSpec) -> PreparedMutation:
        def prepare(cursor: Any) -> PreparedMutation:
            cursor.execute(
                sql.SQL(
                    "SELECT key, generation, locator, expected_lineage, expected_revision, "
                    "expected_generation, expected_manifest_digest, manifest "
                    "FROM {} WHERE operation_id = %s"
                ).format(self._table("mutations")),
                (spec.operation_id,),
            )
            existing = cursor.fetchone()
            expected_values = (
                spec.key, spec.generation, spec.candidate_locator, spec.expected.lineage,
                spec.expected.revision, spec.expected.generation, spec.expected.manifest_digest,
                spec.manifest,
            )
            if existing is not None:
                if tuple(existing) == expected_values:
                    return PreparedMutation(spec.operation_id, spec)
                raise CacheBlobLifecycleConflictError("Operation identifier is not reusable")
            observed = self._expectation(cursor, spec.key)
            if not self._matches(spec.expected, observed):
                raise CacheBlobLifecycleConflictError(
                    "Mutation expectation no longer matches authority"
                )
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {} (operation_id, key, generation, locator, expected_lineage, "
                    "expected_revision, expected_generation, expected_manifest_digest, manifest, "
                    "verified_digest, verified_size, state) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, 'prepared')"
                ).format(self._table("mutations")),
                (
                    spec.operation_id, spec.key, spec.generation, spec.candidate_locator,
                    spec.expected.lineage, spec.expected.revision, spec.expected.generation,
                    spec.expected.manifest_digest, spec.manifest,
                ),
            )
            return PreparedMutation(spec.operation_id, spec)

        return self._transaction("prepare_mutation", prepare)

    def record_verification(self, prepared: PreparedMutation, proof: VerificationProof) -> None:
        if prepared.spec.manifest and proof.manifest and prepared.spec.manifest != proof.manifest:
            raise CacheBlobLifecycleConflictError(
                "Verification descriptor differs from prepared descriptor"
            )
        descriptor = proof.manifest or prepared.spec.manifest

        def record(cursor: Any) -> None:
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET verified_digest = %s, verified_size = %s, manifest = %s "
                    "WHERE operation_id = %s AND state = 'prepared' AND manifest = %s "
                    "AND (verified_digest IS NULL OR "
                    "(verified_digest = %s AND verified_size = %s)) RETURNING operation_id"
                ).format(self._table("mutations")),
                (
                    proof.digest, proof.byte_size, descriptor, prepared.operation_id,
                    prepared.spec.manifest, proof.digest, proof.byte_size,
                ),
            )
            if cursor.fetchone() is None:
                raise CacheBlobLifecycleConflictError(
                    "Prepared mutation cannot accept verification"
                )

        self._transaction("record_verification", record)

    def _read_mutation(self, cursor: Any, operation_id: str, *, lock: bool = False) -> _MutationRow | None:
        suffix = " FOR UPDATE" if lock else ""
        cursor.execute(
            sql.SQL(
                "SELECT key, generation, locator, expected_lineage, expected_revision, "
                "expected_generation, expected_manifest_digest, manifest, verified_digest, "
                "verified_size, state FROM {} WHERE operation_id = %s"
            ).format(self._table("mutations")) + sql.SQL(suffix),
            (operation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        spec = MutationSpec.create(
            operation_id=operation_id, key=row[0], generation=row[1], candidate_locator=row[2],
            expected=EntryExpectation(row[3], row[4], row[5], row[6]), manifest=bytes(row[7]),
        )
        return _MutationRow(spec, row[8], row[9], row[10])

    def _promoted_result(self, cursor: Any, operation_id: str) -> PromotionResult:
        cursor.execute(
            sql.SQL(
                "SELECT m.key, e.generation, e.locator, e.manifest, e.manifest_digest, "
                "e.lineage, e.revision FROM {} AS m JOIN {} AS e ON e.key = m.key "
                "WHERE m.operation_id = %s AND m.state = 'promoted'"
            ).format(self._table("mutations"), self._table("entries")),
            (operation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise CacheBlobLifecycleConflictError("Mutation is not promoted")
        manifest = bytes(row[3])
        if hashlib.sha256(manifest).hexdigest() != row[4]:
            raise CacheBlobBackendError(
                "Committed PostgreSQL lifecycle promotion is malformed",
                context={"operation": "postgresql_lifecycle_authority_promote"},
            )
        entry = EntrySnapshot(
            row[0], row[1], row[2], manifest, EntryExpectation(row[5], row[6], row[1], row[4])
        )
        cursor.execute(
            sql.SQL(
                "SELECT debt_id, operation_id, locator, key, generation, role FROM {} "
                "WHERE operation_id = %s AND state = 'pending' ORDER BY debt_id"
            ).format(self._table("cleanup_debt")),
            (operation_id,),
        )
        debts = tuple(
            CleanupDebt(row[1], row[2], row[3], row[4], row[5], row[0])
            for row in cursor.fetchall()
        )
        return PromotionResult(entry, debts)

    def _classify_promoted_mutation(self, prepared: PreparedMutation) -> PromotionResult | None:
        def classify(cursor: Any) -> PromotionResult | None:
            mutation = self._read_mutation(cursor, prepared.operation_id)
            if mutation is not None and mutation.state == "promoted":
                return self._promoted_result(cursor, prepared.operation_id)
            return None

        return self._read_only("classify_uncertain_promotion", classify)

    def promote_mutation(self, prepared: PreparedMutation) -> PromotionResult:
        try:
            return self._transaction(
                "promote_mutation", lambda cursor: self._promote(cursor, prepared)
            )
        except (CacheBlobLifecycleConflictError, CacheBlobMigrationRequiredError):
            raise
        except (CacheBlobBackendError, CacheBlobLifecycleTimeoutError) as error:
            # A failing commit is ambiguous.  A fresh lease may prove this exact
            # operation promoted; otherwise preserve the original typed failure.
            try:
                classified = self._classify_promoted_mutation(prepared)
            except (CacheBlobBackendError, CacheBlobLifecycleTimeoutError):
                classified = None
            if classified is not None:
                return classified
            raise error

    def _promote(self, cursor: Any, prepared: PreparedMutation) -> PromotionResult:
        mutation = self._read_mutation(cursor, prepared.operation_id, lock=True)
        if mutation is None:
            raise CacheBlobLifecycleConflictError("Mutation does not exist")
        if mutation.spec != prepared.spec:
            raise CacheBlobLifecycleConflictError("Prepared mutation identity is incompatible")
        if mutation.state == "promoted":
            return self._promoted_result(cursor, prepared.operation_id)
        if mutation.state != "prepared" or mutation.verified_digest is None:
            raise CacheBlobLifecycleConflictError("Mutation is not verified and prepared")
        cursor.execute(
            sql.SQL(
                "INSERT INTO {} (key, lineage) VALUES (%s, 0) "
                "ON CONFLICT (key) DO NOTHING RETURNING key"
            ).format(self._table("entry_lineage")),
            (mutation.spec.key,),
        )
        created_lineage = cursor.fetchone() is not None
        observed = self._expectation(cursor, mutation.spec.key, lock=True)
        # The row inserted above is the transaction's own fresh lineage
        # sentinel.  It must preserve future ABA evidence without making a
        # first create conflict with its own absent expectation.
        if created_lineage and observed.revision is None:
            observed = EntryExpectation.absent()
        if not self._matches(mutation.spec.expected, observed):
            raise CacheBlobLifecycleConflictError("Mutation lineage changed before promotion")
        cursor.execute(
            sql.SQL("SELECT generation, locator FROM {} WHERE key = %s FOR UPDATE").format(
                self._table("entries")
            ),
            (mutation.spec.key,),
        )
        old_entry = cursor.fetchone()
        cursor.execute(
            sql.SQL(
                "SELECT authority_revision FROM {} WHERE singleton = TRUE FOR UPDATE"
            ).format(self._table("authority_meta"))
        )
        revision_row = cursor.fetchone()
        if revision_row is None:
            raise CacheBlobMigrationRequiredError("PostgreSQL lifecycle authority is absent")
        next_revision = int(revision_row[0]) + 1
        cursor.execute(
            sql.SQL("UPDATE {} SET lineage = lineage + 1 WHERE key = %s RETURNING lineage").format(
                self._table("entry_lineage")
            ),
            (mutation.spec.key,),
        )
        lineage_row = cursor.fetchone()
        if lineage_row is None:
            raise CacheBlobLifecycleConflictError("Entry lineage cannot be promoted")
        next_lineage = int(lineage_row[0])
        digest = hashlib.sha256(mutation.spec.manifest).hexdigest()
        cursor.execute(
            sql.SQL(
                "INSERT INTO {} (key, generation, locator, manifest, manifest_digest, lineage, revision) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (key) DO UPDATE SET generation = EXCLUDED.generation, "
                "locator = EXCLUDED.locator, manifest = EXCLUDED.manifest, "
                "manifest_digest = EXCLUDED.manifest_digest, lineage = EXCLUDED.lineage, "
                "revision = EXCLUDED.revision WHERE {}.lineage IS NOT DISTINCT FROM %s "
                "AND {}.revision IS NOT DISTINCT FROM %s AND {}.generation IS NOT DISTINCT FROM %s "
                "AND {}.manifest_digest IS NOT DISTINCT FROM %s RETURNING generation, locator"
            ).format(
                self._table("entries"), self._table("entries"), self._table("entries"),
                self._table("entries"), self._table("entries"),
            ),
            (
                mutation.spec.key, mutation.spec.generation, mutation.spec.candidate_locator,
                mutation.spec.manifest, digest, next_lineage, next_revision,
                mutation.spec.expected.lineage, mutation.spec.expected.revision,
                mutation.spec.expected.generation, mutation.spec.expected.manifest_digest,
            ),
        )
        if cursor.fetchone() is None:
            raise CacheBlobLifecycleConflictError("Entry compare-and-swap failed during promotion")
        cursor.execute(
            sql.SQL(
                "UPDATE {} SET state = 'promoted' WHERE operation_id = %s AND state = 'prepared' "
                "AND verified_digest = %s RETURNING operation_id"
            ).format(self._table("mutations")),
            (prepared.operation_id, mutation.verified_digest),
        )
        if cursor.fetchone() is None:
            raise CacheBlobLifecycleConflictError("Prepared mutation cannot be promoted")
        if old_entry is not None and old_entry[1] != mutation.spec.candidate_locator:
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {} (operation_id, key, generation, locator, role, state) "
                    "VALUES (%s, %s, %s, %s, 'previous_generation', 'pending') "
                    "ON CONFLICT (operation_id, locator, role) DO NOTHING"
                ).format(self._table("cleanup_debt")),
                (prepared.operation_id, mutation.spec.key, old_entry[0], old_entry[1]),
            )
        cursor.execute(
            sql.SQL(
                "UPDATE {} SET authority_revision = %s, projection_dirty = TRUE "
                "WHERE singleton = TRUE AND authority_revision = %s RETURNING authority_revision"
            ).format(self._table("authority_meta")),
            (next_revision, next_revision - 1),
        )
        if cursor.fetchone() is None:
            raise CacheBlobLifecycleConflictError("Authority revision compare-and-swap failed")
        return self._promoted_result(cursor, prepared.operation_id)

    def abort_mutation(self, prepared: PreparedMutation, *, candidate_persisted: bool = False) -> None:
        def abort(cursor: Any) -> None:
            mutation = self._read_mutation(cursor, prepared.operation_id, lock=True)
            if mutation is None or mutation.state == "promoted":
                return
            if mutation.spec != prepared.spec:
                raise CacheBlobLifecycleConflictError("Prepared mutation identity is incompatible")
            if not candidate_persisted:
                cursor.execute(
                    sql.SQL("DELETE FROM {} WHERE operation_id = %s AND state <> 'promoted'").format(
                        self._table("mutations")
                    ),
                    (prepared.operation_id,),
                )
                return
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET state = 'aborted' WHERE operation_id = %s "
                    "AND state IN ('prepared', 'aborted') RETURNING operation_id"
                ).format(self._table("mutations")),
                (prepared.operation_id,),
            )
            if cursor.fetchone() is None:
                raise CacheBlobLifecycleConflictError("Mutation cannot be aborted")
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {} (operation_id, key, generation, locator, role, state) "
                    "VALUES (%s, %s, %s, %s, 'candidate', 'pending') "
                    "ON CONFLICT (operation_id, locator, role) DO NOTHING"
                ).format(self._table("cleanup_debt")),
                (
                    prepared.operation_id, mutation.spec.key, mutation.spec.generation,
                    mutation.spec.candidate_locator,
                ),
            )

        self._transaction("abort_mutation", abort)

    def close(self) -> None:
        """Close the authority boundary without closing caller-owned pools."""
        self._closed = True


__all__ = [
    "POSTGRESQL_AUTHORITY_CAPABILITY",
    "POSTGRESQL_AUTHORITY_SCHEMA_VERSION",
    "PostgresqlLifecycleAuthority",
    "SCHEMA_VERSION",
]
