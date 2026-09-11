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
import sys
from typing import Any, Callable, Iterator, TypeVar
import uuid

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobMigrationOfflineDecisionRequiredError,
    CacheBlobMigrationRequiredError,
    CacheBlobStoreClosedError,
)
from cacheness.storage.lifecycle_authority import (
    AuthorityCapabilities,
    AuthorityStateSnapshot,
    CleanupDebt,
    EntryExpectation,
    EntrySnapshot,
    MutationReplay,
    MutationSpec,
    PageToken,
    PreparedMutation,
    ProjectionBackup,
    ProjectionRevision,
    PromotionResult,
    ReconciliationPage,
    ReconciliationSnapshot,
    ReconciliationWork,
    VerificationProof,
)
from cacheness.storage.migration_authority import (
    ActivationReceipt,
    AuthorityInventoryCursor,
    AuthorityInventoryEntry,
    AuthorityIdentitySnapshot,
    AuthorityInventoryPage,
    AuthorityPublicationState,
    FinalizeReceipt,
    PriorStoreReceipt,
    RollbackReceipt,
    VerifiedCandidateReceipt,
    candidate_digest,
    validate_inventory_page_request,
)
from cacheness.storage.manifest import BlobManifest
from cacheness.storage.catalog import (
    CatalogCursor,
    CatalogCursorError,
    CatalogPage,
    CatalogQuery,
    CatalogSchema,
    page_from_canonical_scan,
    validate_catalog_page_request,
)

try:  # Keep the package importable without the optional PostgreSQL extra.
    from psycopg import errors, sql
except ImportError:  # pragma: no cover - exercised by optional-dependency users.
    errors = None
    sql = None


POSTGRESQL_AUTHORITY_SCHEMA_VERSION = 4
"""First-release PostgreSQL authority layout; development schema 3 is unsupported."""

POSTGRESQL_AUTHORITY_CAPABILITY = "postgresql-lifecycle-authority-v4"
"""Persisted authority capability marker, independent of payload formats."""

SCHEMA_VERSION = POSTGRESQL_AUTHORITY_SCHEMA_VERSION
"""Short diagnostic alias for the current persisted PostgreSQL schema only."""

_SCHEMA_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}\Z")
_MAX_STORE_IDENTITY_BYTES = 64
_MAX_AUTHORITY_TEXT_BYTES = 512
_PROGRESS_SQLSTATE_OUTCOMES = {
    "40001": "serialization",
    "40P01": "deadlock",
    "55P03": "lock_timeout",
    "57014": "statement_timeout",
}
_RETRYABLE_SQLSTATES = frozenset(_PROGRESS_SQLSTATE_OUTCOMES)
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
        "migration_store_entries",
    }
)
_REQUIRED_CONSTRAINTS = frozenset(
    {
        "authority_meta_singleton_check",
        "mutations_operation_id_key",
        "cleanup_debt_operation_locator_role_key",
        "mutations_mutation_id_key",
        "clear_targets_run_id_key_key",
        "reconciliation_actions_run_id_source_action_id_key",
        "migration_store_entries_run_id_selection_key_key",
    }
)
_T = TypeVar("_T")


def _bounded_identity(value: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_STORE_IDENTITY_BYTES:
        raise ValueError("store_identity must be a non-empty bounded string")
    return value


def _bounded_authority_text(value: str, field_name: str) -> str:
    """Validate persisted authority identifiers before exposing semantic values."""
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > _MAX_AUTHORITY_TEXT_BYTES
    ):
        raise ValueError(f"{field_name} must be a non-empty bounded string")
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
    allowed_progress_outcomes = frozenset(
        {
            "success",
            "conflict",
            "retryable_serialization",
            "retryable_deadlock",
            "retryable_lock_timeout",
            "retryable_statement_timeout",
            "retryable_connection_timeout",
        }
    )
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
        lifecycle_limits: LifecycleLimits | None = None,
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
        self.lifecycle_limits = (
            LifecycleLimits() if lifecycle_limits is None else lifecycle_limits
        )
        if not isinstance(self.lifecycle_limits, LifecycleLimits):
            raise TypeError("lifecycle_limits must be a LifecycleLimits instance")
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

    @staticmethod
    def _token_value(token: PageToken) -> str:
        if not isinstance(token, PageToken):
            raise TypeError("token must be a PageToken")
        return _bounded_authority_text(token.value, "token")

    @contextmanager
    def _lease(self) -> Iterator[Any]:
        """Lease, then release, exactly one caller-owned connection resource."""
        self._require_open()
        resource = self._connection_factory()
        connection: Any | None = None
        entered = False
        exit_type: type[BaseException] | None = None
        exit_value: BaseException | None = None
        exit_traceback: Any = None
        try:
            if hasattr(resource, "__enter__") and hasattr(resource, "__exit__"):
                connection = resource.__enter__()
                entered = True
            else:
                connection = resource
            if connection is None or not hasattr(connection, "cursor"):
                raise TypeError("connection_factory must return a psycopg connection or lease")
            yield connection
        except BaseException:
            exit_type, exit_value, exit_traceback = sys.exc_info()
            raise
        finally:
            if entered:
                resource.__exit__(exit_type, exit_value, exit_traceback)
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

    @staticmethod
    def progress_outcome_for_sqlstate(sqlstate: str) -> str | None:
        """Return the documented bounded-progress class for one SQLSTATE."""
        return _PROGRESS_SQLSTATE_OUTCOMES.get(sqlstate)

    def _raise_driver_error(self, error: BaseException, *, stage: str) -> None:
        """Map only bounded, non-secret driver context into public exceptions."""
        sqlstate = self._sqlstate(error)
        context = {"operation": "postgresql_lifecycle_authority", "stage": stage}
        progress_outcome = self.progress_outcome_for_sqlstate(sqlstate or "")
        if progress_outcome is not None:
            context["sqlstate"] = sqlstate
            context["progress_outcome"] = progress_outcome
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
            class_outcomes = {
                "SerializationFailure": "serialization",
                "DeadlockDetected": "deadlock",
                "LockNotAvailable": "lock_timeout",
                "QueryCanceled": "statement_timeout",
            }
            context["progress_outcome"] = class_outcomes[error.__class__.__name__]
            raise CacheBlobLifecycleTimeoutError(
                "PostgreSQL lifecycle authority made no bounded progress", context=context
            ) from error
        if sqlstate is None and error.__class__.__name__ in {
            "OperationalError",
            "InterfaceError",
            "ConnectionException",
            "ConnectionTimeout",
        }:
            context["progress_outcome"] = "connection_timeout"
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

    def _inventory_identity(self, cursor: Any) -> AuthorityIdentitySnapshot:
        """Read the persisted remote identity and revision in one transaction."""
        cursor.execute(
            sql.SQL(
                "SELECT schema_version, store_identity, capability, authority_revision "
                "FROM {} WHERE singleton = TRUE"
            ).format(self._table("authority_meta"))
        )
        metadata = cursor.fetchone()
        if metadata is None or len(metadata) != 4:
            raise CacheBlobMigrationRequiredError("PostgreSQL lifecycle authority is absent")
        version, identity, capability, revision = metadata
        try:
            identity = _bounded_identity(identity)
        except ValueError as error:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority store identity is incompatible"
            ) from error
        if version != POSTGRESQL_AUTHORITY_SCHEMA_VERSION:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority schema version is incompatible"
            )
        if capability != POSTGRESQL_AUTHORITY_CAPABILITY:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority capability is incompatible"
            )
        if type(revision) is not int or revision < 0:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL lifecycle authority revision is incompatible"
            )
        self.store_identity = identity
        return AuthorityIdentitySnapshot(
            store_id=identity,
            revision=revision,
            authority_kind="postgresql",
            capability=capability,
            schema_version=version,
        )

    def _known_table_names(self, cursor: Any) -> set[str]:
        """Return only tables belonging to this exact persisted authority layout."""
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = ANY(%s)",
            (self.schema, list(_REQUIRED_TABLES)),
        )
        return {row[0] for row in cursor.fetchall()}

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
        table_names = self._known_table_names(cursor)
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

    def identity_snapshot(self) -> AuthorityIdentitySnapshot:
        """Return persisted PostgreSQL capability, schema, identity, and revision."""
        return self._read_only("migration_identity_snapshot", self._inventory_identity)

    def inventory_page(
        self,
        cursor: AuthorityInventoryCursor | None = None,
        *,
        limit: int | None = None,
        work_cap: int | None = None,
    ) -> AuthorityInventoryPage:
        """Return one raw indexed page without materializing a remote catalog.

        This deterministic contract keeps PostgreSQL's transaction and typed
        progress outcomes intact.  It is not live-service qualification.
        """
        effective_limit, effective_work_cap = validate_inventory_page_request(
            limit=limit,
            work_cap=work_cap,
            default_limit=self.lifecycle_limits.manifest_page_size,
            default_work_cap=self.lifecycle_limits.max_operation_record_bytes,
        )

        def page(read_cursor: Any) -> AuthorityInventoryPage:
            identity = self._inventory_identity(read_cursor)
            if cursor is not None and (
                cursor.store_id != identity.store_id
                or cursor.revision != identity.revision
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration inventory changed; reinspection is required"
                )
            if cursor is None or cursor.last_key is None:
                statement = sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision "
                    "FROM {} ORDER BY key, generation LIMIT %s"
                ).format(self._table("entries"))
                parameters: tuple[Any, ...] = (effective_limit + 1,)
            else:
                statement = sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision "
                    "FROM {} WHERE key > %s OR (key = %s AND generation > %s) "
                    "ORDER BY key, generation LIMIT %s"
                ).format(self._table("entries"))
                parameters = (
                    cursor.last_key,
                    cursor.last_key,
                    cursor.last_generation,
                    effective_limit + 1,
                )
            read_cursor.execute(statement, parameters)
            rows = read_cursor.fetchall()
            entries: list[EntrySnapshot] = []
            work_seen = 0
            for row in rows:
                if len(entries) == effective_limit:
                    break
                entry = self._entry_from_row(row, stage="inventory_page")
                entry_bytes = len(entry.manifest)
                if work_seen + entry_bytes > effective_work_cap:
                    if not entries:
                        raise CacheBlobBackendError(
                            "Migration inventory entry exceeds the configured work bound",
                            context={
                                "operation": "postgresql_lifecycle_authority",
                                "stage": "inventory_page",
                            },
                        )
                    break
                entries.append(entry)
                work_seen += entry_bytes
            exhausted = len(entries) == len(rows) and len(rows) <= effective_limit
            next_cursor = None
            if not exhausted:
                last = entries[-1]
                next_cursor = AuthorityInventoryCursor(
                    store_id=identity.store_id,
                    revision=identity.revision,
                    last_key=last.key,
                    last_generation=last.generation,
                )
            return AuthorityInventoryPage(
                identity=identity,
                entries=tuple(entries),
                next_cursor=next_cursor,
                exhausted=exhausted,
            )

        return self._read_only("inventory_page", page)

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

    @staticmethod
    def _candidate_entries_from_rows(
        rows: list[tuple[Any, ...]],
    ) -> tuple[AuthorityInventoryEntry, ...]:
        """Decode exact retained candidate rows without treating objects as authority."""
        entries: list[AuthorityInventoryEntry] = []
        for row in rows:
            if len(row) != 5:
                raise CacheBlobBackendError(
                    "PostgreSQL migration candidate row is malformed",
                    context={"operation": "migration_candidate"},
                )
            key, generation, locator, manifest_bytes, stored_manifest_digest = row
            try:
                manifest = BlobManifest.from_canonical_bytes(manifest_bytes)
                entry = AuthorityInventoryEntry(
                    key=key,
                    generation=generation,
                    locator=locator,
                    manifest=manifest_bytes,
                    payload_digest=manifest.digest,
                    byte_size=manifest.byte_size,
                )
            except (TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "PostgreSQL migration candidate row is malformed",
                    context={"operation": "migration_candidate"},
                ) from error
            if (
                manifest.key != entry.key
                or manifest.generation != entry.generation
                or manifest.locator != entry.locator
                or manifest.canonical_bytes() != manifest_bytes
                or hashlib.sha256(manifest_bytes).hexdigest() != stored_manifest_digest
            ):
                raise CacheBlobBackendError(
                    "PostgreSQL migration candidate row does not corroborate its manifest",
                    context={"operation": "migration_candidate"},
                )
            entries.append(entry)
        return tuple(entries)

    @staticmethod
    def _validate_verified_candidate(
        identity: AuthorityIdentitySnapshot,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> None:
        """Reject stale, partial, or malformed external evidence before publication."""
        if (
            receipt.destination_identity != identity
            or receipt.destination_revision != identity.revision
        ):
            raise CacheBlobLifecycleConflictError(
                "Migration destination identity or revision changed before candidate recording"
            )
        if len(entries) != receipt.entry_count:
            raise CacheBlobLifecycleConflictError(
                "Migration candidate entry count disagrees with verified receipt"
            )
        if sum(entry.byte_size for entry in entries) != receipt.byte_count:
            raise CacheBlobLifecycleConflictError(
                "Migration candidate byte count disagrees with verified receipt"
            )
        if len({entry.key for entry in entries}) != len(entries):
            raise CacheBlobLifecycleConflictError("Migration candidate contains duplicate keys")
        if candidate_digest(entries) != receipt.candidate_digest:
            raise CacheBlobLifecycleConflictError(
                "Migration candidate digest disagrees with verified receipt"
            )
        for candidate in entries:
            try:
                manifest = BlobManifest.from_canonical_bytes(candidate.manifest)
            except (TypeError, ValueError) as error:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate manifest is malformed"
                ) from error
            if (
                manifest.key != candidate.key
                or manifest.generation != candidate.generation
                or manifest.locator != candidate.locator
                or manifest.digest != candidate.payload_digest
                or manifest.byte_size != candidate.byte_size
                or manifest.canonical_bytes() != candidate.manifest
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate descriptor does not corroborate its receipt"
                )

    def _publication_state_row(
        self, cursor: Any, *, lock: bool = False
    ) -> tuple[Any, ...]:
        """Load the complete authority-owned cutover state in one database transaction."""
        suffix = " FOR UPDATE" if lock else ""
        cursor.execute(
            sql.SQL(
                "SELECT authority_revision, migration_run_id, migration_plan_digest, "
                "migration_candidate_digest, migration_source_revision, "
                "migration_activated_revision, migration_state, migration_active_selection, "
                "migration_rollback_eligible FROM {} WHERE singleton = TRUE"
            ).format(self._table("authority_meta"))
            + sql.SQL(suffix)
        )
        row = cursor.fetchone()
        if row is None or len(row) != 9:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL migration publication state is incompatible"
            )
        try:
            state = AuthorityPublicationState(row[6])
        except (TypeError, ValueError) as error:
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL migration publication state is incompatible"
            ) from error
        if (
            type(row[0]) is not int
            or row[0] < 0
            or row[7] not in {"source", "candidate", "prior"}
            or type(row[8]) is not bool
        ):
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL migration publication state is incompatible"
            )
        if state is AuthorityPublicationState.IDLE and (
            row[1] is not None
            or row[2] is not None
            or row[3] is not None
            or row[4] is not None
            or row[5] is not None
            or row[7] != "source"
            or row[8]
        ):
            raise CacheBlobMigrationRequiredError(
                "PostgreSQL idle migration publication state is incompatible"
            )
        return tuple(row)

    def record_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> VerifiedCandidateReceipt:
        """Persist one complete candidate without selecting it for ordinary workers."""
        if not isinstance(receipt, VerifiedCandidateReceipt) or not isinstance(entries, tuple):
            raise TypeError("migration candidate receipt and entries must be immutable values")

        def record(cursor: Any) -> VerifiedCandidateReceipt:
            identity = self._inventory_identity(cursor)
            self._validate_verified_candidate(identity, receipt, entries)
            state_row = self._publication_state_row(cursor, lock=True)
            state = AuthorityPublicationState(state_row[6])
            if state is AuthorityPublicationState.CANDIDATE:
                if (
                    state_row[1] == receipt.run_id
                    and state_row[2] == receipt.plan_digest
                    and state_row[4] == receipt.source_revision
                ):
                    cursor.execute(
                        sql.SQL(
                            "SELECT key, generation, locator, manifest, manifest_digest "
                            "FROM {} WHERE run_id = %s AND selection = 'candidate' ORDER BY key"
                        ).format(self._table("migration_store_entries")),
                        (receipt.run_id,),
                    )
                    stored = self._candidate_entries_from_rows(cursor.fetchall())
                    if stored == entries:
                        return receipt
                    if stored == entries[: len(stored)]:
                        for entry in entries[len(stored) :]:
                            cursor.execute(
                                sql.SQL(
                                    "INSERT INTO {} (run_id, selection, key, generation, locator, manifest, "
                                    "manifest_digest, lineage, entry_revision) "
                                    "VALUES (%s, 'candidate', %s, %s, %s, %s, %s, 0, %s)"
                                ).format(self._table("migration_store_entries")),
                                (
                                    receipt.run_id,
                                    entry.key,
                                    entry.generation,
                                    entry.locator,
                                    entry.manifest,
                                    entry.manifest_digest,
                                    receipt.source_revision,
                                ),
                            )
                        cursor.execute(
                            sql.SQL(
                                "UPDATE {} SET migration_candidate_digest = %s "
                                "WHERE singleton = TRUE AND migration_run_id = %s "
                                "AND migration_plan_digest = %s AND migration_state = 'candidate'"
                            ).format(self._table("authority_meta")),
                            (receipt.candidate_digest, receipt.run_id, receipt.plan_digest),
                        )
                        return receipt
                raise CacheBlobLifecycleConflictError(
                    "A different verified migration candidate is already recorded"
                )
            if state is not AuthorityPublicationState.IDLE:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate recording requires an unselected active authority"
                )
            for entry in entries:
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} (run_id, selection, key, generation, locator, manifest, "
                        "manifest_digest, lineage, entry_revision) "
                        "VALUES (%s, 'candidate', %s, %s, %s, %s, %s, 0, %s)"
                    ).format(self._table("migration_store_entries")),
                    (
                        receipt.run_id,
                        entry.key,
                        entry.generation,
                        entry.locator,
                        entry.manifest,
                        entry.manifest_digest,
                        receipt.source_revision,
                    ),
                )
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET migration_run_id = %s, migration_plan_digest = %s, "
                    "migration_candidate_digest = %s, migration_source_revision = %s, "
                    "migration_activated_revision = NULL, migration_state = 'candidate', "
                    "migration_active_selection = 'source', migration_rollback_eligible = FALSE "
                    "WHERE singleton = TRUE AND authority_revision = %s AND migration_state = 'idle' "
                    "AND migration_run_id IS NULL RETURNING authority_revision"
                ).format(self._table("authority_meta")),
                (
                    receipt.run_id,
                    receipt.plan_digest,
                    receipt.candidate_digest,
                    receipt.source_revision,
                    state_row[0],
                ),
            )
            if cursor.fetchone() != (state_row[0],):
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate recording lost the authority selection"
                )
            return receipt

        return self._transaction("record_verified_candidate", record)

    def candidate_entries_for_run(self, *, run_id: str) -> tuple[AuthorityInventoryEntry, ...]:
        """Return only this authority's durably attributed candidate descriptors."""
        def read(cursor: Any) -> tuple[AuthorityInventoryEntry, ...]:
            state_row = self._publication_state_row(cursor)
            if state_row[1] != run_id or AuthorityPublicationState(state_row[6]) not in {
                AuthorityPublicationState.CANDIDATE,
                AuthorityPublicationState.ACTIVATED_OFFLINE,
                AuthorityPublicationState.ACTIVE,
                AuthorityPublicationState.ROLLED_BACK,
            }:
                return ()
            cursor.execute(
                sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest "
                    "FROM {} WHERE run_id = %s AND selection = 'candidate' ORDER BY key"
                ).format(self._table("migration_store_entries")),
                (run_id,),
            )
            return self._candidate_entries_from_rows(cursor.fetchall())

        return self._read_only("candidate_entries_for_run", read)

    def discard_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> None:
        """Clear one exact unactivated candidate after external retirement succeeds."""
        if not isinstance(receipt, VerifiedCandidateReceipt) or not isinstance(entries, tuple):
            raise TypeError("migration candidate receipt and entries must be immutable values")

        def discard(cursor: Any) -> None:
            identity = self._inventory_identity(cursor)
            self._validate_verified_candidate(identity, receipt, entries)
            state_row = self._publication_state_row(cursor, lock=True)
            if (
                AuthorityPublicationState(state_row[6]) is not AuthorityPublicationState.CANDIDATE
                or state_row[1] != receipt.run_id
                or state_row[2] != receipt.plan_digest
                or state_row[3] != receipt.candidate_digest
            ):
                raise CacheBlobLifecycleConflictError(
                    "Only the exact unactivated migration candidate may be discarded"
                )
            cursor.execute(
                sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest "
                    "FROM {} WHERE run_id = %s AND selection = 'candidate' ORDER BY key"
                ).format(self._table("migration_store_entries")),
                (receipt.run_id,),
            )
            if self._candidate_entries_from_rows(cursor.fetchall()) != entries:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate entries changed before discard"
                )
            cursor.execute(
                sql.SQL(
                    "DELETE FROM {} WHERE run_id = %s AND selection = 'candidate'"
                ).format(self._table("migration_store_entries")),
                (receipt.run_id,),
            )
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET migration_run_id = NULL, migration_plan_digest = NULL, "
                    "migration_candidate_digest = NULL, migration_source_revision = NULL, "
                    "migration_activated_revision = NULL, migration_state = 'idle', "
                    "migration_active_selection = 'source', migration_rollback_eligible = FALSE "
                    "WHERE singleton = TRUE AND migration_run_id = %s AND migration_state = 'candidate'"
                ).format(self._table("authority_meta")),
                (receipt.run_id,),
            )
            if cursor.rowcount != 1:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate discard lost the authority selection"
                )

        self._transaction("discard_verified_candidate", discard)

    def activate_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> ActivationReceipt:
        """Make one recorded candidate visible in one PostgreSQL-only transaction."""
        if not isinstance(receipt, VerifiedCandidateReceipt) or not isinstance(entries, tuple):
            raise TypeError("migration candidate receipt and entries must be immutable values")

        def activate(cursor: Any) -> ActivationReceipt:
            state_row = self._publication_state_row(cursor, lock=True)
            state = AuthorityPublicationState(state_row[6])
            if state is AuthorityPublicationState.ACTIVATED_OFFLINE:
                if (
                    state_row[1] == receipt.run_id
                    and state_row[2] == receipt.plan_digest
                    and state_row[3] == receipt.candidate_digest
                    and state_row[4] == receipt.source_revision
                    and type(state_row[5]) is int
                    and state_row[7] == "candidate"
                    and state_row[8] is True
                ):
                    cursor.execute(
                        sql.SQL(
                            "SELECT count(*) FROM {} WHERE run_id = %s AND selection = 'prior'"
                        ).format(self._table("migration_store_entries")),
                        (receipt.run_id,),
                    )
                    count_row = cursor.fetchone()
                    if count_row is None or type(count_row[0]) is not int:
                        raise CacheBlobBackendError(
                            "PostgreSQL retained prior row count is malformed",
                            context={"operation": "migration_activation"},
                        )
                    return ActivationReceipt(
                        candidate_receipt=receipt,
                        activation_revision=state_row[5],
                        prior_store=PriorStoreReceipt(
                            run_id=receipt.run_id,
                            revision=state_row[0] - 1,
                            entry_count=count_row[0],
                        ),
                    )
                raise CacheBlobLifecycleConflictError(
                    "A different migration activation is already selected"
                )
            if (
                state is not AuthorityPublicationState.CANDIDATE
                or state_row[0] != receipt.destination_revision
                or state_row[1] != receipt.run_id
                or state_row[2] != receipt.plan_digest
                or state_row[3] != receipt.candidate_digest
                or state_row[4] != receipt.source_revision
                or state_row[5] is not None
                or state_row[7] != "source"
                or state_row[8] is not False
            ):
                raise CacheBlobLifecycleConflictError(
                    "Verified migration candidate is not recorded by this authority"
                )
            cursor.execute(
                sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest "
                    "FROM {} WHERE run_id = %s AND selection = 'candidate' ORDER BY key"
                ).format(self._table("migration_store_entries")),
                (receipt.run_id,),
            )
            if self._candidate_entries_from_rows(cursor.fetchall()) != entries:
                raise CacheBlobLifecycleConflictError(
                    "Recorded migration candidate differs from verified external receipt"
                )
            cursor.execute(
                sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision "
                    "FROM {} ORDER BY key"
                ).format(self._table("entries"))
            )
            prior_rows = cursor.fetchall()
            for prior in prior_rows:
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} (run_id, selection, key, generation, locator, manifest, "
                        "manifest_digest, lineage, entry_revision) "
                        "VALUES (%s, 'prior', %s, %s, %s, %s, %s, %s, %s)"
                    ).format(self._table("migration_store_entries")),
                    (receipt.run_id, *prior),
                )
            next_revision = state_row[0] + 1
            cursor.execute(sql.SQL("DELETE FROM {}").format(self._table("entries")))
            for candidate in entries:
                cursor.execute(
                    sql.SQL("SELECT lineage FROM {} WHERE key = %s").format(
                        self._table("entry_lineage")
                    ),
                    (candidate.key,),
                )
                prior_lineage = cursor.fetchone()
                lineage = (prior_lineage[0] if prior_lineage is not None else 0) + 1
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} (key, lineage) VALUES (%s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET lineage = EXCLUDED.lineage"
                    ).format(self._table("entry_lineage")),
                    (candidate.key, lineage),
                )
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} (key, generation, locator, manifest, manifest_digest, "
                        "lineage, revision) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    ).format(self._table("entries")),
                    (
                        candidate.key,
                        candidate.generation,
                        candidate.locator,
                        candidate.manifest,
                        candidate.manifest_digest,
                        lineage,
                        next_revision,
                    ),
                )
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET authority_revision = %s, projection_dirty = TRUE, "
                    "migration_activated_revision = %s, migration_state = 'activated_offline', "
                    "migration_active_selection = 'candidate', migration_rollback_eligible = TRUE "
                    "WHERE singleton = TRUE AND authority_revision = %s AND migration_state = 'candidate' "
                    "AND migration_run_id = %s RETURNING authority_revision"
                ).format(self._table("authority_meta")),
                (next_revision, next_revision, state_row[0], receipt.run_id),
            )
            if cursor.fetchone() != (next_revision,):
                raise CacheBlobLifecycleConflictError(
                    "Migration activation lost the authority selection"
                )
            return ActivationReceipt(
                candidate_receipt=receipt,
                activation_revision=next_revision,
                prior_store=PriorStoreReceipt(
                    run_id=receipt.run_id,
                    revision=state_row[0],
                    entry_count=len(prior_rows),
                ),
            )

        return self._transaction("activate_verified_candidate", activate)

    def publication_state(self) -> AuthorityPublicationState:
        """Read the authority-owned maintenance state with no payload inference."""
        # BlobStore invokes the ordinary-worker fence before its explicit
        # initialization boundary.  A constructor has no persisted authority
        # state to inspect yet, so it is safely idle until ``initialize``
        # performs the real schema/version validation.
        if self.store_identity is None:
            return AuthorityPublicationState.IDLE
        return self._read_only(
            "publication_state",
            lambda cursor: AuthorityPublicationState(self._publication_state_row(cursor)[6]),
        )

    def activation_receipt_for_candidate(
        self, receipt: VerifiedCandidateReceipt
    ) -> ActivationReceipt | None:
        """Classify a lost response only from exact persisted authority state."""
        if not isinstance(receipt, VerifiedCandidateReceipt):
            raise TypeError("receipt must be a VerifiedCandidateReceipt")

        def receipt_for_candidate(cursor: Any) -> ActivationReceipt | None:
            state_row = self._publication_state_row(cursor)
            if (
                AuthorityPublicationState(state_row[6])
                is not AuthorityPublicationState.ACTIVATED_OFFLINE
                or state_row[1] != receipt.run_id
                or state_row[2] != receipt.plan_digest
                or state_row[3] != receipt.candidate_digest
                or state_row[4] != receipt.source_revision
                or type(state_row[5]) is not int
            ):
                return None
            cursor.execute(
                sql.SQL(
                    "SELECT count(*) FROM {} WHERE run_id = %s AND selection = 'prior'"
                ).format(self._table("migration_store_entries")),
                (receipt.run_id,),
            )
            count_row = cursor.fetchone()
            if count_row is None or type(count_row[0]) is not int:
                raise CacheBlobBackendError(
                    "PostgreSQL retained prior row count is malformed",
                    context={"operation": "migration_activation"},
                )
            return ActivationReceipt(
                candidate_receipt=receipt,
                activation_revision=state_row[5],
                prior_store=PriorStoreReceipt(
                    run_id=receipt.run_id,
                    revision=state_row[0] - 1,
                    entry_count=count_row[0],
                ),
            )

        return self._read_only("activation_receipt_for_candidate", receipt_for_candidate)

    def require_ordinary_worker_access(self) -> None:
        """Fence ordinary BlobStore work during the explicit rollback window."""
        if self.publication_state() is AuthorityPublicationState.ACTIVATED_OFFLINE:
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "Offline migration activation requires explicit rollback or finalize before workers restart",
                context={"operation": "migration.worker_access"},
            )

    def rollback_verified_candidate(self, *, run_id: str) -> RollbackReceipt:
        """Restore retained prior rows in one authority transaction while workers stop."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")

        def rollback(cursor: Any) -> RollbackReceipt:
            state_row = self._publication_state_row(cursor, lock=True)
            if (
                AuthorityPublicationState(state_row[6])
                is not AuthorityPublicationState.ACTIVATED_OFFLINE
                or state_row[1] != run_id
                or state_row[8] is not True
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration rollback requires the selected offline activation"
                )
            cursor.execute(
                sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest, lineage "
                    "FROM {} WHERE run_id = %s AND selection = 'prior' ORDER BY key"
                ).format(self._table("migration_store_entries")),
                (run_id,),
            )
            prior_rows = cursor.fetchall()
            next_revision = state_row[0] + 1
            cursor.execute(sql.SQL("DELETE FROM {}").format(self._table("entries")))
            for prior in prior_rows:
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} (key, lineage) VALUES (%s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET lineage = EXCLUDED.lineage"
                    ).format(self._table("entry_lineage")),
                    (prior[0], prior[5]),
                )
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} (key, generation, locator, manifest, manifest_digest, "
                        "lineage, revision) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    ).format(self._table("entries")),
                    (*prior, next_revision),
                )
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET authority_revision = %s, projection_dirty = TRUE, "
                    "migration_state = 'rolled_back', migration_active_selection = 'prior', "
                    "migration_rollback_eligible = FALSE WHERE singleton = TRUE "
                    "AND authority_revision = %s AND migration_state = 'activated_offline' "
                    "AND migration_run_id = %s RETURNING authority_revision"
                ).format(self._table("authority_meta")),
                (next_revision, state_row[0], run_id),
            )
            if cursor.fetchone() != (next_revision,):
                raise CacheBlobLifecycleConflictError(
                    "Migration rollback lost the authority selection"
                )
            return RollbackReceipt(run_id=run_id, rollback_revision=next_revision)

        return self._transaction("rollback_verified_candidate", rollback)

    def finalize_verified_candidate(self, *, run_id: str) -> FinalizeReceipt:
        """End rollback eligibility without deleting separately retained prior data."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")

        def finalize(cursor: Any) -> FinalizeReceipt:
            state_row = self._publication_state_row(cursor, lock=True)
            if (
                AuthorityPublicationState(state_row[6])
                is not AuthorityPublicationState.ACTIVATED_OFFLINE
                or state_row[1] != run_id
                or state_row[8] is not True
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration finalize requires the selected offline activation"
                )
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET migration_state = 'active', migration_active_selection = 'candidate', "
                    "migration_rollback_eligible = FALSE WHERE singleton = TRUE "
                    "AND authority_revision = %s AND migration_state = 'activated_offline' "
                    "AND migration_run_id = %s RETURNING authority_revision"
                ).format(self._table("authority_meta")),
                (state_row[0], run_id),
            )
            if cursor.fetchone() != (state_row[0],):
                raise CacheBlobLifecycleConflictError(
                    "Migration finalize lost the authority selection"
                )
            return FinalizeReceipt(run_id=run_id, finalized_revision=state_row[0])

        return self._transaction("finalize_verified_candidate", finalize)

    def _initialize_once(self, cursor: Any) -> None:
        """Emit current-version DDL only in the caller-selected initialization action."""
        # Existing recognized authority tables are evidence of a persisted
        # layout.  Validate them before DDL so a partial/foreign schema is
        # never completed, adopted, or implicitly migrated.
        if self._known_table_names(cursor):
            self._validate_schema(cursor)
            return
        schema = sql.Identifier(self.schema)
        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "singleton BOOLEAN PRIMARY KEY DEFAULT TRUE, "
                "schema_version INTEGER NOT NULL, store_identity TEXT NOT NULL, "
                "capability TEXT NOT NULL, authority_revision BIGINT NOT NULL, "
                "projection_dirty BOOLEAN NOT NULL, migration_run_id TEXT, "
                "migration_plan_digest TEXT, migration_candidate_digest TEXT, "
                "migration_source_revision BIGINT, migration_activated_revision BIGINT, "
                "migration_state TEXT NOT NULL DEFAULT 'idle' "
                "CHECK (migration_state IN ('idle', 'candidate', 'activated_offline', 'active', 'rolled_back')), "
                "migration_active_selection TEXT NOT NULL DEFAULT 'source' "
                "CHECK (migration_active_selection IN ('source', 'candidate', 'prior')), "
                "migration_rollback_eligible BOOLEAN NOT NULL DEFAULT FALSE, "
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
                "mutation_id BIGSERIAL UNIQUE NOT NULL, operation_id TEXT PRIMARY KEY, "
                "key TEXT NOT NULL, generation TEXT NOT NULL, "
                "locator TEXT NOT NULL, expected_lineage BIGINT, expected_revision BIGINT, "
                "expected_generation TEXT, expected_manifest_digest TEXT, manifest BYTEA NOT NULL, "
                "verified_digest TEXT, verified_size BIGINT, state TEXT NOT NULL, "
                "promoted_lineage BIGINT, promoted_revision BIGINT, "
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
                "capture_cursor TEXT NOT NULL DEFAULT '', snapshot_complete BOOLEAN NOT NULL DEFAULT FALSE, "
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
                "run_id TEXT NOT NULL, source TEXT NOT NULL, action_id BIGINT NOT NULL, state TEXT NOT NULL, "
                "CONSTRAINT reconciliation_actions_run_id_source_action_id_key "
                "PRIMARY KEY (run_id, source, action_id))"
            ).format(self._table("reconciliation_actions"))
        )
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "run_id TEXT NOT NULL, selection TEXT NOT NULL "
                "CHECK (selection IN ('candidate', 'prior')), key TEXT NOT NULL, "
                "generation TEXT NOT NULL, locator TEXT NOT NULL, manifest BYTEA NOT NULL, "
                "manifest_digest TEXT NOT NULL, lineage BIGINT NOT NULL, "
                "entry_revision BIGINT NOT NULL, "
                "CONSTRAINT migration_store_entries_run_id_selection_key_key "
                "PRIMARY KEY (run_id, selection, key))"
            ).format(self._table("migration_store_entries"))
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
        self.require_ordinary_worker_access()

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
            return self._entry_from_row((key, *row), stage="read_entry")

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
            try:
                states = tuple(
                    (
                        _bounded_authority_text(row[0], "operation_id"),
                        _bounded_authority_text(row[1], "mutation_state"),
                    )
                    for row in cursor.fetchall()
                )
            except (TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "PostgreSQL lifecycle authority mutation row is malformed",
                    context={"operation": "postgresql_lifecycle_authority", "stage": "snapshot_state"},
                ) from error
            cursor.execute(
                sql.SQL(
                    "SELECT debt_id, operation_id, locator, key, generation, role FROM {} "
                    "WHERE state = 'pending' ORDER BY debt_id LIMIT %s"
                ).format(self._table("cleanup_debt")),
                (self.lifecycle_limits.operation_page_size + 1,),
            )
            debt_rows = cursor.fetchall()
            if len(debt_rows) > self.lifecycle_limits.operation_page_size:
                raise CacheBlobBackendError(
                    "PostgreSQL lifecycle authority diagnostic debt page exceeds its bound",
                    context={"operation": "postgresql_lifecycle_authority", "stage": "snapshot_state"},
                )
            debts = tuple(
                CleanupDebt(row[1], row[2], row[3], row[4], row[5], row[0])
                for row in debt_rows
            )
            return AuthorityStateSnapshot(int(meta[0]), bool(meta[1]), states, debts)

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

    def read_mutation(self, operation_id: str) -> MutationReplay | None:
        """Read one exact operation replay from the existing remote authority."""
        MutationSpec.validate_operation_id(operation_id)

        def read(cursor: Any) -> MutationReplay | None:
            try:
                mutation = self._read_mutation(cursor, operation_id)
                if mutation is None:
                    return None
                if (mutation.verified_digest is None) != (mutation.verified_size is None):
                    raise ValueError("verification proof fields are incomplete")
                proof = (
                    None
                    if mutation.verified_digest is None
                    else VerificationProof(
                        mutation.verified_digest,
                        mutation.verified_size,
                        mutation.spec.manifest,
                    )
                )
                prepared = PreparedMutation(operation_id, mutation.spec)
                if mutation.state == "prepared":
                    return MutationReplay(prepared, mutation.state, proof)
                if mutation.state == "promoted":
                    return MutationReplay(
                        prepared,
                        mutation.state,
                        proof,
                        self._promoted_result(cursor, operation_id),
                    )
            except (TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "PostgreSQL lifecycle authority mutation row is malformed",
                    context={
                        "operation": "postgresql_lifecycle_authority",
                        "stage": "read_mutation",
                    },
                ) from error
            raise CacheBlobLifecycleConflictError(
                "PostgreSQL lifecycle operation is not replayable",
                context={"operation_id": operation_id, "state": mutation.state},
            )

        return self._read_only("read_mutation", read)

    def _entry_from_row(self, row: tuple[Any, ...], *, stage: str) -> EntrySnapshot:
        """Decode one canonical row only after its bounded values corroborate."""
        key, generation, locator, manifest, digest, lineage, revision = row
        try:
            manifest = bytes(manifest)
        except (TypeError, ValueError) as error:
            raise CacheBlobBackendError(
                "PostgreSQL lifecycle authority entry row is malformed",
                context={"operation": "postgresql_lifecycle_authority", "stage": stage},
            ) from error
        if len(manifest) > self.lifecycle_limits.max_operation_record_bytes:
            raise CacheBlobBackendError(
                "PostgreSQL lifecycle authority entry exceeds the configured record bound",
                context={"operation": "postgresql_lifecycle_authority", "stage": stage},
            )
        if hashlib.sha256(manifest).hexdigest() != digest:
            raise CacheBlobBackendError(
                "PostgreSQL lifecycle authority entry row is malformed",
                context={"operation": "postgresql_lifecycle_authority", "stage": stage},
            )
        try:
            return EntrySnapshot(
                key,
                generation,
                locator,
                manifest,
                EntryExpectation(lineage, revision, generation, digest),
            )
        except (TypeError, ValueError) as error:
            raise CacheBlobBackendError(
                "PostgreSQL lifecycle authority entry row is malformed",
                context={"operation": "postgresql_lifecycle_authority", "stage": stage},
            ) from error

    def _promoted_result(self, cursor: Any, operation_id: str) -> PromotionResult:
        """Reconstruct an idempotent receipt from immutable operation evidence."""
        cursor.execute(
            sql.SQL(
                "SELECT key, generation, locator, manifest, promoted_lineage, "
                "promoted_revision FROM {} "
                "WHERE operation_id = %s AND state = 'promoted'"
            ).format(self._table("mutations")),
            (operation_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise CacheBlobLifecycleConflictError("Mutation is not promoted")
        try:
            manifest = bytes(row[3])
            manifest_digest = hashlib.sha256(manifest).hexdigest()
            entry = self._entry_from_row(
                (row[0], row[1], row[2], manifest, manifest_digest, row[4], row[5]),
                stage="promote_mutation",
            )
        except (TypeError, ValueError) as error:
            raise CacheBlobBackendError(
                "PostgreSQL promoted mutation receipt is malformed",
                context={
                    "operation": "postgresql_lifecycle_authority",
                    "stage": "promote_mutation",
                },
            ) from error
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
                "UPDATE {} SET state = 'promoted', promoted_lineage = %s, "
                "promoted_revision = %s WHERE operation_id = %s AND state = 'prepared' "
                "AND verified_digest = %s RETURNING operation_id"
            ).format(self._table("mutations")),
            (
                next_lineage,
                next_revision,
                prepared.operation_id,
                mutation.verified_digest,
            ),
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

    def list_entries(self) -> tuple[EntrySnapshot, ...]:
        """Return one explicitly bounded diagnostic page of committed entries.

        Remote callers use :meth:`catalog_page`; this legacy-shaped method is
        intentionally capped so it cannot become an unbounded remote scan.
        """

        def list_page(cursor: Any) -> tuple[EntrySnapshot, ...]:
            cursor.execute(
                sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision "
                    "FROM {} ORDER BY key, generation LIMIT %s"
                ).format(self._table("entries")),
                (self.lifecycle_limits.max_inventory_items + 1,),
            )
            rows = cursor.fetchall()
            if len(rows) > self.lifecycle_limits.max_inventory_items:
                raise CacheBlobBackendError(
                    "PostgreSQL lifecycle authority list_entries exceeds its bounded diagnostic page",
                    context={"operation": "postgresql_lifecycle_authority", "stage": "list_entries"},
                )
            return tuple(self._entry_from_row(row, stage="list_entries") for row in rows)

        return self._read_only("list_entries", list_page)

    def catalog_page(
        self,
        query: CatalogQuery,
        cursor: str | None,
        *,
        schema: CatalogSchema,
        limit: int,
        work_cap: int,
        signing_key: bytes,
        manifest_loader: Callable[[bytes], Any],
    ) -> CatalogPage:
        """Scan only one canonical, revision-bound keyset page in PostgreSQL."""
        validate_catalog_page_request(
            query,
            schema=schema,
            cursor=cursor,
            limit=limit,
            work_cap=work_cap,
        )
        if cursor is not None:
            CatalogCursor.inspect(cursor, signing_key=signing_key)

        def page(read_cursor: Any) -> CatalogPage:
            metadata = self._metadata(read_cursor)
            if metadata is None:
                raise CacheBlobMigrationRequiredError("PostgreSQL lifecycle authority is absent")
            _version, store_id, _capability = metadata
            read_cursor.execute(
                sql.SQL("SELECT authority_revision FROM {} WHERE singleton = TRUE").format(
                    self._table("authority_meta")
                )
            )
            revision_row = read_cursor.fetchone()
            if revision_row is None or type(revision_row[0]) is not int:
                raise CacheBlobMigrationRequiredError(
                    "PostgreSQL lifecycle authority revision is incompatible"
                )
            revision = revision_row[0]
            cursor_identity = (
                None
                if cursor is None
                else CatalogCursor.parse(
                    cursor,
                    store_id=store_id,
                    format_version=2,
                    schema_id=schema.schema_id,
                    schema_fingerprint=schema.fingerprint,
                    query_fingerprint=query.fingerprint,
                    revision=revision,
                    signing_key=signing_key,
                )
            )
            if cursor_identity is None:
                statement = sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision "
                    "FROM {} ORDER BY key, generation LIMIT %s"
                ).format(self._table("entries"))
                parameters: tuple[Any, ...] = (work_cap + 1,)
            else:
                statement = sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision "
                    "FROM {} WHERE key > %s OR (key = %s AND generation > %s) "
                    "ORDER BY key, generation LIMIT %s"
                ).format(self._table("entries"))
                parameters = (
                    cursor_identity[0],
                    cursor_identity[0],
                    cursor_identity[1],
                    work_cap + 1,
                )
            read_cursor.execute(statement, parameters)
            snapshots = tuple(
                self._entry_from_row(row, stage="catalog_page")
                for row in read_cursor.fetchall()
            )
            return page_from_canonical_scan(
                snapshots,
                query=query,
                schema=schema,
                revision=revision,
                store_id=store_id,
                cursor_identity=cursor_identity,
                limit=limit,
                work_cap=work_cap,
                signing_key=signing_key,
                manifest_loader=manifest_loader,
            )

        try:
            return self._read_only("catalog_page", page)
        except CatalogCursorError:
            raise

    def pending_cleanup_debts(
        self,
        *,
        key: str | None = None,
        operation_id: str | None = None,
    ) -> tuple[CleanupDebt, ...]:
        """Return only one bounded, exact cleanup-debt work page."""
        if key is not None:
            _bounded_authority_text(key, "key")
        if operation_id is not None:
            _bounded_authority_text(operation_id, "operation_id")

        def read_debts(cursor: Any) -> tuple[CleanupDebt, ...]:
            predicates = [sql.SQL("state = 'pending'")]
            values: list[Any] = []
            if key is not None:
                predicates.append(sql.SQL("key = %s"))
                values.append(key)
            if operation_id is not None:
                predicates.append(sql.SQL("operation_id = %s"))
                values.append(operation_id)
            statement = (
                sql.SQL(
                    "SELECT debt_id, operation_id, locator, key, generation, role FROM {} WHERE "
                ).format(self._table("cleanup_debt"))
                + sql.SQL(" AND ").join(predicates)
                + sql.SQL(" ORDER BY debt_id LIMIT %s")
            )
            values.append(self.lifecycle_limits.operation_page_size + 1)
            cursor.execute(statement, tuple(values))
            rows = cursor.fetchall()
            if len(rows) > self.lifecycle_limits.operation_page_size:
                raise CacheBlobBackendError(
                    "PostgreSQL cleanup debt page exceeds its configured bound",
                    context={"operation": "postgresql_lifecycle_authority", "stage": "pending_cleanup_debts"},
                )
            try:
                return tuple(
                    CleanupDebt(row[1], row[2], row[3], row[4], row[5], row[0])
                    for row in rows
                )
            except (TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "PostgreSQL cleanup debt row is malformed",
                    context={"operation": "postgresql_lifecycle_authority", "stage": "pending_cleanup_debts"},
                ) from error

        return self._read_only("pending_cleanup_debts", read_debts)

    def pending_mutations(self) -> tuple[PreparedMutation, ...]:
        """Return one bounded recovery page of exactly prepared operations."""

        def read_mutations(cursor: Any) -> tuple[PreparedMutation, ...]:
            cursor.execute(
                sql.SQL(
                    "SELECT operation_id, key, generation, locator, expected_lineage, "
                    "expected_revision, expected_generation, expected_manifest_digest, manifest "
                    "FROM {} WHERE state = 'prepared' ORDER BY mutation_id LIMIT %s"
                ).format(self._table("mutations")),
                (self.lifecycle_limits.operation_page_size + 1,),
            )
            rows = cursor.fetchall()
            if len(rows) > self.lifecycle_limits.operation_page_size:
                raise CacheBlobBackendError(
                    "PostgreSQL mutation page exceeds its configured bound",
                    context={"operation": "postgresql_lifecycle_authority", "stage": "pending_mutations"},
                )
            try:
                return tuple(
                    PreparedMutation(
                        row[0],
                        MutationSpec.create(
                            operation_id=row[0],
                            key=row[1],
                            generation=row[2],
                            candidate_locator=row[3],
                            expected=EntryExpectation(row[4], row[5], row[6], row[7]),
                            manifest=bytes(row[8]),
                        ),
                    )
                    for row in rows
                )
            except (TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "PostgreSQL mutation row is malformed",
                    context={"operation": "postgresql_lifecycle_authority", "stage": "pending_mutations"},
                ) from error

        return self._read_only("pending_mutations", read_mutations)

    def retire_cleanup_debt(self, debt: CleanupDebt) -> None:
        """Idempotently retire only the exact debt proven externally reclaimed."""

        def retire(cursor: Any) -> None:
            predicates = [
                sql.SQL("operation_id = %s"),
                sql.SQL("locator = %s"),
                sql.SQL("key = %s"),
                sql.SQL("generation = %s"),
                sql.SQL("role = %s"),
                sql.SQL("state = 'pending'"),
            ]
            values: list[Any] = [
                debt.operation_id,
                debt.locator,
                debt.key,
                debt.generation,
                debt.role,
            ]
            if debt.debt_id is not None:
                predicates.insert(0, sql.SQL("debt_id = %s"))
                values.insert(0, debt.debt_id)
            statement = (
                sql.SQL("DELETE FROM {} WHERE ").format(self._table("cleanup_debt"))
                + sql.SQL(" AND ").join(predicates)
                + sql.SQL(" RETURNING debt_id")
            )
            cursor.execute(statement, tuple(values))
            cursor.fetchone()

        self._transaction("retire_cleanup_debt", retire)

    def delete_entry(self, key: str, *, expected: EntryExpectation) -> None:
        """Retire one exact visible generation and advance its lineage atomically."""
        _bounded_authority_text(key, "key")

        def delete(cursor: Any) -> None:
            observed = self._expectation(cursor, key, lock=True)
            if not self._matches(expected, observed):
                raise CacheBlobLifecycleConflictError(
                    "Delete expectation no longer matches authority"
                )
            cursor.execute(
                sql.SQL("INSERT INTO {} (key, lineage) VALUES (%s, 0) ON CONFLICT (key) DO NOTHING").format(
                    self._table("entry_lineage")
                ),
                (key,),
            )
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE key = %s").format(self._table("entries")),
                (key,),
            )
            cursor.execute(
                sql.SQL("UPDATE {} SET lineage = lineage + 1 WHERE key = %s RETURNING lineage").format(
                    self._table("entry_lineage")
                ),
                (key,),
            )
            if cursor.fetchone() is None:
                raise CacheBlobLifecycleConflictError("Entry lineage cannot be retired")
            cursor.execute(
                sql.SQL("SELECT authority_revision FROM {} WHERE singleton = TRUE FOR UPDATE").format(
                    self._table("authority_meta")
                )
            )
            revision_row = cursor.fetchone()
            if revision_row is None:
                raise CacheBlobMigrationRequiredError("PostgreSQL lifecycle authority is absent")
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET authority_revision = %s, projection_dirty = TRUE "
                    "WHERE singleton = TRUE AND authority_revision = %s RETURNING authority_revision"
                ).format(self._table("authority_meta")),
                (int(revision_row[0]) + 1, revision_row[0]),
            )
            if cursor.fetchone() is None:
                raise CacheBlobLifecycleConflictError("Authority revision compare-and-swap failed")

        self._transaction("delete_entry", delete)

    def retire_tombstone(self, key: str, *, expected: EntryExpectation) -> None:
        """Retire tombstone-shaped descriptors through the exact deletion primitive."""
        self.delete_entry(key, expected=expected)

    def begin_clear(self) -> PageToken:
        """Durably begin or resume a revision-bounded clear snapshot."""

        def begin(cursor: Any) -> PageToken:
            cursor.execute(
                sql.SQL(
                    "SELECT run_id FROM {} WHERE state = 'active' ORDER BY run_id LIMIT 1 FOR UPDATE"
                ).format(self._table("clear_runs"))
            )
            active = cursor.fetchone()
            if active is not None:
                try:
                    return PageToken(_bounded_authority_text(active[0], "run_id"))
                except (TypeError, ValueError) as error:
                    raise CacheBlobBackendError(
                        "PostgreSQL clear run row is malformed",
                        context={"operation": "postgresql_lifecycle_authority", "stage": "begin_clear"},
                    ) from error
            cursor.execute(
                sql.SQL("SELECT authority_revision FROM {} WHERE singleton = TRUE FOR UPDATE").format(
                    self._table("authority_meta")
                )
            )
            revision = cursor.fetchone()
            if revision is None:
                raise CacheBlobMigrationRequiredError("PostgreSQL lifecycle authority is absent")
            token = PageToken(uuid.uuid4().hex)
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {} (run_id, state, revision, capture_cursor, snapshot_complete, last_key) "
                    "VALUES (%s, 'active', %s, '', FALSE, '')"
                ).format(self._table("clear_runs")),
                (token.value, revision[0]),
            )
            return token

        return self._transaction("begin_clear", begin)

    def _capture_clear_page(self, cursor: Any, token: PageToken) -> None:
        """Persist one finite membership page before returning its targets."""
        token_value = self._token_value(token)
        cursor.execute(
            sql.SQL(
                "SELECT revision, capture_cursor, snapshot_complete FROM {} "
                "WHERE run_id = %s AND state = 'active' FOR UPDATE"
            ).format(self._table("clear_runs")),
            (token_value,),
        )
        run = cursor.fetchone()
        if run is None:
            raise CacheBlobLifecycleConflictError("Clear run does not exist")
        if bool(run[2]):
            return
        cursor.execute(
            sql.SQL(
                "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision "
                "FROM {} WHERE revision <= %s AND key > %s ORDER BY key LIMIT %s"
            ).format(self._table("entries")),
            (run[0], run[1], self.lifecycle_limits.manifest_page_size + 1),
        )
        rows = cursor.fetchall()
        selected = rows[: self.lifecycle_limits.manifest_page_size]
        for row in selected:
            entry = self._entry_from_row(row, stage="capture_clear_page")
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {} (run_id, key, lineage, entry_revision, generation, locator, manifest, "
                    "manifest_digest, state) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending') "
                    "ON CONFLICT (run_id, key) DO NOTHING"
                ).format(self._table("clear_targets")),
                (
                    token_value,
                    entry.key,
                    entry.expectation.lineage,
                    entry.expectation.revision,
                    entry.generation,
                    entry.locator,
                    entry.manifest,
                    entry.expectation.manifest_digest,
                ),
            )
        capture_cursor = selected[-1][0] if selected else run[1]
        cursor.execute(
            sql.SQL(
                "UPDATE {} SET capture_cursor = %s, snapshot_complete = %s WHERE run_id = %s"
            ).format(self._table("clear_runs")),
            (capture_cursor, len(rows) <= self.lifecycle_limits.manifest_page_size, token_value),
        )

    def page_clear(self, token: PageToken) -> tuple[EntrySnapshot, ...]:
        """Return a bounded persisted clear-target page, extending the snapshot safely."""
        token_value = self._token_value(token)

        def page(cursor: Any) -> tuple[EntrySnapshot, ...]:
            self._capture_clear_page(cursor, token)
            cursor.execute(
                sql.SQL(
                    "SELECT c.state, c.last_key FROM {} AS c WHERE c.run_id = %s"
                ).format(self._table("clear_runs")),
                (token_value,),
            )
            run = cursor.fetchone()
            if run is None:
                raise CacheBlobLifecycleConflictError("Clear run does not exist")
            if run[0] != "active":
                return ()
            cursor.execute(
                sql.SQL(
                    "SELECT key, generation, locator, manifest, manifest_digest, lineage, entry_revision "
                    "FROM {} WHERE run_id = %s AND state = 'pending' AND key > %s "
                    "ORDER BY key LIMIT %s"
                ).format(self._table("clear_targets")),
                (token_value, run[1], self.lifecycle_limits.manifest_page_size + 1),
            )
            rows = cursor.fetchall()
            selected = rows[: self.lifecycle_limits.manifest_page_size]
            bytes_seen = 0
            entries: list[EntrySnapshot] = []
            for row in selected:
                entry = self._entry_from_row(row, stage="page_clear")
                if bytes_seen + len(entry.manifest) > self.lifecycle_limits.max_operation_record_bytes:
                    if not entries:
                        raise CacheBlobBackendError(
                            "PostgreSQL clear target exceeds the configured work bound",
                            context={"operation": "postgresql_lifecycle_authority", "stage": "page_clear"},
                        )
                    break
                entries.append(entry)
                bytes_seen += len(entry.manifest)
            return tuple(entries)

        return self._transaction("page_clear", page)

    def checkpoint_clear(
        self,
        token: PageToken,
        target: EntrySnapshot | None = None,
        *,
        state: str = "completed",
    ) -> None:
        """Checkpoint exact clear work without ever deriving membership from S3."""
        token_value = self._token_value(token)

        def checkpoint(cursor: Any) -> None:
            cursor.execute(
                sql.SQL(
                    "SELECT state, snapshot_complete FROM {} WHERE run_id = %s FOR UPDATE"
                ).format(self._table("clear_runs")),
                (token_value,),
            )
            run = cursor.fetchone()
            if target is None and run is not None and run[0] == "completed":
                return
            if run is None or run[0] != "active":
                raise CacheBlobLifecycleConflictError("Clear run cannot accept checkpoint")
            if target is None:
                if not bool(run[1]):
                    raise CacheBlobLifecycleConflictError("Clear snapshot is not complete")
                cursor.execute(
                    sql.SQL(
                        "UPDATE {} SET state = 'completed' WHERE run_id = %s AND NOT EXISTS "
                        "(SELECT 1 FROM {} WHERE run_id = %s AND state = 'pending') RETURNING run_id"
                    ).format(self._table("clear_runs"), self._table("clear_targets")),
                    (token_value, token_value),
                )
                if cursor.fetchone() is None:
                    raise CacheBlobLifecycleConflictError(
                        "Clear run cannot complete while targets remain"
                    )
                return
            if state not in {"completed", "conflicted", "blocked"}:
                raise ValueError("Clear target state is unsupported")
            cursor.execute(
                sql.SQL(
                    "SELECT state FROM {} WHERE run_id = %s AND key = %s AND lineage = %s "
                    "AND entry_revision = %s AND generation = %s AND manifest_digest = %s FOR UPDATE"
                ).format(self._table("clear_targets")),
                (
                    token_value,
                    target.key,
                    target.expectation.lineage,
                    target.expectation.revision,
                    target.generation,
                    target.expectation.manifest_digest,
                ),
            )
            row = cursor.fetchone()
            if row is None or row[0] != "pending":
                raise CacheBlobLifecycleConflictError("Clear target is no longer pending")
            cursor.execute(
                sql.SQL(
                    "SELECT lineage, revision, generation, manifest_digest FROM {} WHERE key = %s"
                ).format(self._table("entries")),
                (target.key,),
            )
            current = cursor.fetchone()
            if state == "completed":
                cursor.execute(
                    sql.SQL("SELECT lineage FROM {} WHERE key = %s").format(
                        self._table("entry_lineage")
                    ),
                    (target.key,),
                )
                lineage = cursor.fetchone()
                if current is not None or lineage is None or lineage[0] <= target.expectation.lineage:
                    raise CacheBlobLifecycleConflictError(
                        "Clear target completion lacks exact absence proof"
                    )
            elif state == "conflicted" and current == (
                target.expectation.lineage,
                target.expectation.revision,
                target.generation,
                target.expectation.manifest_digest,
            ):
                raise CacheBlobLifecycleConflictError("Clear target has not changed")
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET state = %s WHERE run_id = %s AND key = %s AND state = 'pending' "
                    "RETURNING key"
                ).format(self._table("clear_targets")),
                (state, token_value, target.key),
            )
            if cursor.fetchone() is None:
                raise CacheBlobLifecycleConflictError("Clear target cannot accept checkpoint")
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET last_key = %s, state = CASE WHEN snapshot_complete AND NOT EXISTS "
                    "(SELECT 1 FROM {} WHERE run_id = %s AND state = 'pending') THEN 'completed' "
                    "ELSE 'active' END WHERE run_id = %s"
                ).format(self._table("clear_runs"), self._table("clear_targets")),
                (target.key, token_value, token_value),
            )

        self._transaction("checkpoint_clear", checkpoint)

    def begin_reconciliation(self) -> PageToken:
        """Create or resume one durable high-water reconciliation run."""

        def begin(cursor: Any) -> PageToken:
            cursor.execute(
                sql.SQL(
                    "SELECT run_id FROM {} WHERE state = 'active' ORDER BY run_id LIMIT 1 FOR UPDATE"
                ).format(self._table("reconciliation_runs"))
            )
            active = cursor.fetchone()
            if active is not None:
                try:
                    return PageToken(_bounded_authority_text(active[0], "run_id"))
                except (TypeError, ValueError) as error:
                    raise CacheBlobBackendError(
                        "PostgreSQL reconciliation run row is malformed",
                        context={"operation": "postgresql_lifecycle_authority", "stage": "begin_reconciliation"},
                    ) from error
            cursor.execute(sql.SQL("SELECT COALESCE(MAX(mutation_id), 0) FROM {}").format(self._table("mutations")))
            mutation_high_water = cursor.fetchone()[0]
            cursor.execute(sql.SQL("SELECT COALESCE(MAX(debt_id), 0) FROM {}").format(self._table("cleanup_debt")))
            debt_high_water = cursor.fetchone()[0]
            cursor.execute(
                sql.SQL("SELECT authority_revision FROM {} WHERE singleton = TRUE FOR UPDATE").format(
                    self._table("authority_meta")
                )
            )
            revision = cursor.fetchone()
            if revision is None:
                raise CacheBlobMigrationRequiredError("PostgreSQL lifecycle authority is absent")
            token = PageToken(uuid.uuid4().hex)
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {} (run_id, state, mutation_high_water, debt_high_water, authority_revision, "
                    "mutation_cursor, debt_cursor) VALUES (%s, 'active', %s, %s, %s, 0, 0)"
                ).format(self._table("reconciliation_runs")),
                (token.value, mutation_high_water, debt_high_water, revision[0]),
            )
            return token

        return self._transaction("begin_reconciliation", begin)

    def reconciliation_snapshot(
        self, token: PageToken | None = None
    ) -> ReconciliationSnapshot:
        """Return durable high-water bounds without payload inspection."""
        token_value = None if token is None else self._token_value(token)

        def snapshot(cursor: Any) -> ReconciliationSnapshot:
            if token_value is not None:
                cursor.execute(
                    sql.SQL(
                        "SELECT authority_revision, mutation_high_water, debt_high_water FROM {} "
                        "WHERE run_id = %s"
                    ).format(self._table("reconciliation_runs")),
                    (token_value,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise CacheBlobLifecycleConflictError("Reconciliation run does not exist")
                return ReconciliationSnapshot(row[0], row[1], row[2], token_value)
            cursor.execute(
                sql.SQL("SELECT authority_revision FROM {} WHERE singleton = TRUE").format(
                    self._table("authority_meta")
                )
            )
            revision = cursor.fetchone()
            if revision is None:
                raise CacheBlobMigrationRequiredError("PostgreSQL lifecycle authority is absent")
            cursor.execute(sql.SQL("SELECT COALESCE(MAX(mutation_id), 0) FROM {}").format(self._table("mutations")))
            mutation_high_water = cursor.fetchone()[0]
            cursor.execute(sql.SQL("SELECT COALESCE(MAX(debt_id), 0) FROM {}").format(self._table("cleanup_debt")))
            debt_high_water = cursor.fetchone()[0]
            return ReconciliationSnapshot(revision[0], mutation_high_water, debt_high_water)

        return self._read_only("reconciliation_snapshot", snapshot)

    def inventory_locator_attribution(
        self,
        snapshot: ReconciliationSnapshot,
        locators: tuple[str, ...],
    ) -> frozenset[str] | None:
        """Return only locators owned by one exact reconciliation snapshot.

        S3 inventory is evidence, never a visibility authority.  This bounded
        lookup lets the reconciler distinguish a committed locator from an
        unknown one without scanning the catalog or treating an S3 key as
        provenance.  One SQL statement gives the revision check and the three
        ownership sources the same PostgreSQL MVCC snapshot.  A concurrent
        authority revision makes the page indeterminate rather than silently
        attributing it to a different lifecycle state.
        """
        if not isinstance(snapshot, ReconciliationSnapshot):
            raise TypeError("snapshot must be a ReconciliationSnapshot")
        if not isinstance(locators, tuple):
            raise TypeError("locators must be a tuple")
        if len(locators) > self.lifecycle_limits.max_inventory_items:
            raise ValueError("inventory locator page exceeds the configured bound")
        normalized = tuple(
            _bounded_authority_text(locator, "inventory locator") for locator in locators
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("inventory locator page contains duplicate locators")
        if not normalized:
            return frozenset()

        def locate(cursor: Any) -> frozenset[str] | None:
            # A single statement is intentional: PostgreSQL takes one MVCC
            # snapshot per statement at READ COMMITTED.  Splitting this into a
            # revision read and a locator query could mix two lifecycle states.
            cursor.execute(
                sql.SQL(
                    "WITH revision AS ("
                    "SELECT authority_revision FROM {} WHERE singleton = TRUE"
                    "), owned AS ("
                    "SELECT locator FROM {} WHERE locator = ANY(%s) "
                    "UNION SELECT locator FROM {} "
                    "WHERE state = 'prepared' AND mutation_id <= %s AND locator = ANY(%s) "
                    "UNION SELECT locator FROM {} "
                    "WHERE state = 'pending' AND debt_id <= %s AND locator = ANY(%s)"
                    ") SELECT revision.authority_revision, owned.locator "
                    "FROM revision LEFT JOIN owned ON TRUE"
                ).format(
                    self._table("authority_meta"),
                    self._table("entries"),
                    self._table("mutations"),
                    self._table("cleanup_debt"),
                ),
                (
                    list(normalized),
                    snapshot.mutation_high_water,
                    list(normalized),
                    snapshot.debt_high_water,
                    list(normalized),
                ),
            )
            rows = cursor.fetchall()
            if not rows:
                raise CacheBlobMigrationRequiredError(
                    "PostgreSQL lifecycle authority is absent"
                )
            try:
                revisions = {row[0] for row in rows}
            except (IndexError, TypeError) as error:
                raise CacheBlobBackendError(
                    "PostgreSQL inventory attribution row is malformed",
                    context={
                        "operation": "postgresql_lifecycle_authority",
                        "stage": "inventory_locator_attribution",
                    },
                ) from error
            if not all(type(revision) is int and revision >= 0 for revision in revisions):
                raise CacheBlobBackendError(
                    "PostgreSQL inventory attribution row is malformed",
                    context={
                        "operation": "postgresql_lifecycle_authority",
                        "stage": "inventory_locator_attribution",
                    },
                )
            if revisions != {snapshot.authority_revision}:
                return None
            try:
                owned = frozenset(
                    _bounded_authority_text(row[1], "owned inventory locator")
                    for row in rows
                    if row[1] is not None
                )
            except (IndexError, TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "PostgreSQL inventory attribution row is malformed",
                    context={
                        "operation": "postgresql_lifecycle_authority",
                        "stage": "inventory_locator_attribution",
                    },
                ) from error
            if not owned.issubset(normalized):
                raise CacheBlobBackendError(
                    "PostgreSQL inventory attribution escaped its requested page",
                    context={
                        "operation": "postgresql_lifecycle_authority",
                        "stage": "inventory_locator_attribution",
                    },
                )
            return owned

        return self._read_only("inventory_locator_attribution", locate)

    def page_reconciliation_work(
        self,
        snapshot: ReconciliationSnapshot,
        *,
        mutation_cursor: int,
        debt_cursor: int,
    ) -> ReconciliationPage:
        """Return one bounded, independent-keyset residue page."""
        if mutation_cursor < 0 or debt_cursor < 0:
            raise ValueError("Reconciliation cursors must be non-negative")

        def page(cursor: Any) -> ReconciliationPage:
            page_size = max(1, self.lifecycle_limits.operation_page_size // 2)
            cursor.execute(
                sql.SQL(
                    "SELECT mutation_id, operation_id, key, generation, locator, expected_lineage, "
                    "expected_revision, expected_generation, expected_manifest_digest, manifest, state "
                    "FROM {} WHERE mutation_id > %s AND mutation_id <= %s AND state = 'prepared' "
                    "ORDER BY mutation_id LIMIT %s"
                ).format(self._table("mutations")),
                (mutation_cursor, snapshot.mutation_high_water, page_size),
            )
            mutation_rows = cursor.fetchall()
            # Keep the normal recovery page strictly pending-only, but surface
            # the earliest unexpected durable state before a later pending row
            # can advance the independent keyset cursor past it.
            cursor.execute(
                sql.SQL(
                    "SELECT debt_id, operation_id, locator, key, generation, role, state FROM {} "
                    "WHERE debt_id > %s AND debt_id <= %s AND state <> 'pending' "
                    "ORDER BY debt_id LIMIT 1"
                ).format(self._table("cleanup_debt")),
                (debt_cursor, snapshot.debt_high_water),
            )
            invalid_debt_row = cursor.fetchone()
            pending_debt_high_water = (
                snapshot.debt_high_water
                if invalid_debt_row is None
                else invalid_debt_row[0] - 1
            )
            cursor.execute(
                sql.SQL(
                    "SELECT debt_id, operation_id, locator, key, generation, role, state FROM {} "
                    "WHERE debt_id > %s AND debt_id <= %s AND state = 'pending' "
                    "ORDER BY debt_id LIMIT %s"
                ).format(self._table("cleanup_debt")),
                (debt_cursor, pending_debt_high_water, page_size),
            )
            debt_rows = cursor.fetchall()
            works: list[ReconciliationWork] = []
            bytes_seen = 0
            # A cursor advances only after the corresponding durable row is
            # emitted.  Fetching ahead is allowed for bounded SQL pages, but
            # never authorizes reconciliation to skip evidence omitted by the
            # shared byte/action work limits.
            next_mutation = (
                snapshot.mutation_high_water if not mutation_rows else mutation_cursor
            )
            for row in mutation_rows:
                manifest = bytes(row[9])
                if len(manifest) > self.lifecycle_limits.max_operation_record_bytes:
                    raise CacheBlobBackendError(
                        "PostgreSQL reconciliation mutation exceeds the configured record bound",
                        context={"operation": "postgresql_lifecycle_authority", "stage": "page_reconciliation_work"},
                    )
                if bytes_seen + len(manifest) > self.lifecycle_limits.max_operation_record_bytes:
                    break
                works.append(
                    ReconciliationWork(
                        "mutation",
                        row[0],
                        row[10],
                        mutation=PreparedMutation(
                            row[1],
                            MutationSpec.create(
                                operation_id=row[1],
                                key=row[2],
                                generation=row[3],
                                candidate_locator=row[4],
                                expected=EntryExpectation(row[5], row[6], row[7], row[8]),
                                manifest=manifest,
                            ),
                        ),
                    )
                )
                bytes_seen += len(manifest)
                next_mutation = row[0]
            next_debt = (
                snapshot.debt_high_water
                if not debt_rows and invalid_debt_row is None
                else debt_cursor
            )
            for row in debt_rows:
                if len(works) >= self.lifecycle_limits.operation_page_size:
                    break
                works.append(
                    ReconciliationWork(
                        "debt",
                        row[0],
                        row[6],
                        debt=CleanupDebt(row[1], row[2], row[3], row[4], row[5], row[0]),
                    )
                )
                next_debt = row[0]
            if (
                invalid_debt_row is not None
                and len(works) < self.lifecycle_limits.operation_page_size
            ):
                row = invalid_debt_row
                works.append(
                    ReconciliationWork(
                        "debt",
                        row[0],
                        row[6],
                        debt=CleanupDebt(row[1], row[2], row[3], row[4], row[5], row[0]),
                    )
                )
                next_debt = row[0]
            return ReconciliationPage(tuple(works), next_mutation, next_debt)

        return self._read_only("page_reconciliation_work", page)

    def page_reconciliation(self, token: PageToken) -> tuple[CleanupDebt, ...]:
        """Return the bounded pending-debt compatibility view for one run."""
        token_value = self._token_value(token)

        def page(cursor: Any) -> tuple[CleanupDebt, ...]:
            cursor.execute(
                sql.SQL("SELECT debt_high_water FROM {} WHERE run_id = %s").format(
                    self._table("reconciliation_runs")
                ),
                (token_value,),
            )
            run = cursor.fetchone()
            if run is None:
                raise CacheBlobLifecycleConflictError("Reconciliation run does not exist")
            cursor.execute(
                sql.SQL(
                    "SELECT debt_id, operation_id, locator, key, generation, role FROM {} "
                    "WHERE state = 'pending' AND debt_id <= %s ORDER BY debt_id LIMIT %s"
                ).format(self._table("cleanup_debt")),
                (run[0], self.lifecycle_limits.operation_page_size + 1),
            )
            rows = cursor.fetchall()
            if len(rows) > self.lifecycle_limits.operation_page_size:
                raise CacheBlobBackendError(
                    "PostgreSQL reconciliation debt page exceeds its configured bound",
                    context={"operation": "postgresql_lifecycle_authority", "stage": "page_reconciliation"},
                )
            return tuple(
                CleanupDebt(row[1], row[2], row[3], row[4], row[5], row[0])
                for row in rows
            )

        return self._read_only("page_reconciliation", page)

    def checkpoint_reconciliation(
        self,
        token: PageToken,
        work: ReconciliationWork | None = None,
        *,
        state: str = "completed",
    ) -> None:
        """Persist a bounded replay checkpoint; retry policy stays with the caller."""
        token_value = self._token_value(token)

        def checkpoint(cursor: Any) -> None:
            if work is None:
                cursor.execute(
                    sql.SQL(
                        "UPDATE {} SET state = 'completed' WHERE run_id = %s AND state = 'active' "
                        "RETURNING run_id"
                    ).format(self._table("reconciliation_runs")),
                    (token_value,),
                )
                if cursor.fetchone() is None:
                    raise CacheBlobLifecycleConflictError(
                        "Reconciliation run cannot accept checkpoint"
                    )
                return
            if state not in {"completed", "blocked", "conflicted"}:
                raise ValueError("Reconciliation checkpoint state is unsupported")
            if work.source not in {"mutation", "debt"}:
                raise ValueError("Reconciliation work source is unsupported")
            cursor_column = "mutation_cursor" if work.source == "mutation" else "debt_cursor"
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {} (run_id, source, action_id, state) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (run_id, source, action_id) DO UPDATE SET state = EXCLUDED.state"
                ).format(self._table("reconciliation_actions")),
                (token_value, work.source, work.row_id, state),
            )
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET {} = GREATEST({}, %s) WHERE run_id = %s AND state = 'active' "
                    "RETURNING run_id"
                ).format(
                    self._table("reconciliation_runs"),
                    sql.Identifier(cursor_column),
                    sql.Identifier(cursor_column),
                ),
                (work.row_id, token_value),
            )
            if cursor.fetchone() is None:
                raise CacheBlobLifecycleConflictError(
                    "Reconciliation run cannot accept checkpoint"
                )

        self._transaction("checkpoint_reconciliation", checkpoint)

    def compare_and_mark_projection(
        self, expected: ProjectionRevision | None
    ) -> ProjectionRevision:
        """Mark only an exact authoritative revision as projected and current."""

        def mark(cursor: Any) -> ProjectionRevision:
            cursor.execute(
                sql.SQL(
                    "SELECT authority_revision, projection_dirty FROM {} WHERE singleton = TRUE FOR UPDATE"
                ).format(self._table("authority_meta"))
            )
            row = cursor.fetchone()
            if row is None:
                raise CacheBlobMigrationRequiredError("PostgreSQL lifecycle authority is absent")
            revision, dirty = row
            if expected is not None and expected.value != revision:
                raise CacheBlobLifecycleConflictError("Projection revision changed")
            if dirty:
                cursor.execute(
                    sql.SQL(
                        "UPDATE {} SET projection_dirty = FALSE WHERE singleton = TRUE AND "
                        "authority_revision = %s RETURNING authority_revision"
                    ).format(self._table("authority_meta")),
                    (revision,),
                )
                if cursor.fetchone() is None:
                    raise CacheBlobLifecycleConflictError("Projection revision changed")
            return ProjectionRevision(revision)

        return self._transaction("compare_and_mark_projection", mark)

    @contextmanager
    def projection_backup(self) -> Iterator[ProjectionBackup]:
        """Fail closed: remote authorities expose paged descriptors, never a local DB file.

        A local SQLite backup would manufacture a second projection authority and
        require an unbounded remote dump.  PostgreSQL projections must consume
        revision-bound :meth:`catalog_page` results instead.
        """
        self._require_open()
        raise CacheBlobBackendError(
            "PostgreSQL lifecycle authority has no local projection backup; use catalog pages",
            context={"operation": "postgresql_lifecycle_authority", "stage": "projection_backup"},
        )
        yield  # pragma: no cover - preserves context-manager typing.

    def close(self) -> None:
        """Close the authority boundary without closing caller-owned pools."""
        self._closed = True


__all__ = [
    "POSTGRESQL_AUTHORITY_CAPABILITY",
    "POSTGRESQL_AUTHORITY_SCHEMA_VERSION",
    "PostgresqlLifecycleAuthority",
    "SCHEMA_VERSION",
]
