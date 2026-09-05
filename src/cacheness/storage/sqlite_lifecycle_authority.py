"""SQLite-backed local transactional authority for BlobStore lifecycle state."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import uuid4

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationRequiredError,
)

from .lifecycle_authority import (
    AuthorityCapabilities,
    CleanupDebt,
    EntryExpectation,
    EntrySnapshot,
    MutationSpec,
    PageToken,
    PreparedMutation,
    ProjectionRevision,
    PromotionResult,
    VerificationProof,
)


AUTHORITY_RELATIVE_PATH = Path(".cacheness") / "lifecycle-authority-v1.sqlite3"
SQLITE_APPLICATION_ID = 0x43414348
SCHEMA_VERSION = 1


class SqliteLifecycleAuthority:
    """One short SQLite transaction owns each committed manifest transition."""

    capabilities = AuthorityCapabilities(durable=True, multiprocess=True)

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.path = self.root / AUTHORITY_RELATIVE_PATH
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = None
        self.open_write_transactions = 0
        self._closed = False

    @classmethod
    def for_root(cls, root: Path | str) -> "SqliteLifecycleAuthority":
        """Return a non-materializing authority; mutation creates its database."""
        return cls(root)

    def _classify_for_open(self) -> str:
        try:
            self.root.lstat()
        except FileNotFoundError:
            return "empty"
        if not self.root.is_dir() or self.root.is_symlink():
            return "wrong_root"
        try:
            entries = tuple(self.root.iterdir())
        except OSError as exc:
            raise CacheBlobMigrationRequiredError("Authority root cannot be inspected") from exc
        if not entries:
            return "empty"
        if self.path.exists():
            if not self.path.is_file() or self.path.is_symlink():
                return "invalid_authority"
            return "authority"
        return "established"

    def _open(self, *, mutation: bool) -> sqlite3.Connection | None:
        if self._closed:
            raise RuntimeError("Lifecycle authority is closed")
        state = self._classify_for_open()
        if state == "empty" and not mutation:
            return None
        if state != "authority" and state != "empty":
            raise CacheBlobMigrationRequiredError(
                "Store has evidence but no valid lifecycle authority",
                context={"authority_path": str(self.path), "classification": state},
            )
        if self._connection is None:
            if state == "empty":
                if os.name == "nt":
                    raise CacheBlobMigrationRequiredError(
                        "Windows lifecycle roots must be provisioned before mutation",
                        context={"authority_path": str(self.path)},
                    )
                self.path.parent.mkdir(parents=True, exist_ok=False)
                connection = sqlite3.connect(self.path, isolation_level=None, timeout=2.0)
                self._initialize_schema(connection)
            else:
                connection = sqlite3.connect(self.path, isolation_level=None, timeout=2.0)
                self._validate_schema(connection)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            self._connection = connection
        return self._connection

    @staticmethod
    def _initialize_schema(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = EXTRA")
        connection.execute("PRAGMA application_id = 1128350536")
        connection.execute("PRAGMA user_version = 1")
        connection.executescript(
            """
            CREATE TABLE store_identity (identity TEXT NOT NULL);
            CREATE TABLE entry_lineage (key TEXT PRIMARY KEY, lineage INTEGER NOT NULL);
            CREATE TABLE entries (
                key TEXT PRIMARY KEY,
                generation TEXT NOT NULL,
                locator TEXT NOT NULL,
                manifest BLOB NOT NULL,
                lineage INTEGER NOT NULL,
                revision INTEGER NOT NULL
            );
            CREATE TABLE mutations (
                operation_id TEXT PRIMARY KEY,
                key TEXT NOT NULL,
                generation TEXT NOT NULL,
                locator TEXT NOT NULL,
                expected_lineage INTEGER,
                expected_revision INTEGER,
                manifest BLOB NOT NULL,
                verified_digest TEXT,
                verified_size INTEGER,
                state TEXT NOT NULL
            );
            INSERT INTO store_identity(identity) VALUES (lower(hex(randomblob(16))));
            """
        )

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        identity = connection.execute("SELECT identity FROM store_identity LIMIT 1").fetchone()
        if application_id != SQLITE_APPLICATION_ID or version != SCHEMA_VERSION or identity is None:
            raise CacheBlobMigrationRequiredError("Lifecycle authority identity is incompatible")

    def _transaction(self, callback):
        connection = self._open(mutation=True)
        assert connection is not None
        with self._lock:
            self.open_write_transactions += 1
            try:
                connection.execute("BEGIN IMMEDIATE")
                result = callback(connection)
                connection.execute("COMMIT")
                return result
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                self.open_write_transactions -= 1

    @staticmethod
    def _expectation(connection: sqlite3.Connection, key: str) -> EntryExpectation:
        row = connection.execute(
            "SELECT lineage, revision FROM entries WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            return EntryExpectation(lineage=row[0], revision=row[1])
        row = connection.execute("SELECT lineage FROM entry_lineage WHERE key = ?", (key,)).fetchone()
        return EntryExpectation(lineage=None if row is None else row[0], revision=None)

    @staticmethod
    def _matches(expected: EntryExpectation, observed: EntryExpectation) -> bool:
        return expected == observed

    def read_entry(self, key: str) -> EntrySnapshot | None:
        connection = self._open(mutation=False)
        if connection is None:
            return None
        row = connection.execute(
            "SELECT generation, locator, manifest, lineage, revision FROM entries WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return EntrySnapshot(key, row[0], row[1], bytes(row[2]), EntryExpectation(row[3], row[4]))

    def prepare_mutation(self, spec: MutationSpec) -> PreparedMutation:
        def prepare(connection: sqlite3.Connection) -> PreparedMutation:
            existing = connection.execute(
                "SELECT state FROM mutations WHERE operation_id = ?", (spec.operation_id,)
            ).fetchone()
            if existing is not None:
                if existing[0] == "prepared":
                    return PreparedMutation(spec.operation_id, spec)
                raise CacheBlobLifecycleConflictError("Operation identifier is not reusable")
            observed = self._expectation(connection, spec.key)
            if not self._matches(spec.expected, observed):
                raise CacheBlobLifecycleConflictError("Mutation expectation no longer matches authority")
            connection.execute(
                "INSERT INTO mutations VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, 'prepared')",
                (spec.operation_id, spec.key, spec.generation, spec.candidate_locator,
                 spec.expected.lineage, spec.expected.revision, spec.manifest),
            )
            return PreparedMutation(spec.operation_id, spec)
        return self._transaction(prepare)

    def record_verification(self, prepared: PreparedMutation, proof: VerificationProof) -> None:
        def record(connection: sqlite3.Connection) -> None:
            cursor = connection.execute(
                "UPDATE mutations SET verified_digest = ?, verified_size = ?, manifest = ? WHERE operation_id = ? AND state = 'prepared'",
                (proof.digest, proof.byte_size, proof.manifest or prepared.spec.manifest, prepared.operation_id),
            )
            if cursor.rowcount != 1:
                raise CacheBlobLifecycleConflictError("Prepared mutation cannot accept verification")
        self._transaction(record)

    def record_verification_for_test(self, prepared: PreparedMutation) -> None:
        """Install deterministic proof for authority-only contract tests."""
        self.record_verification(prepared, VerificationProof("0" * 64, 0))

    def promote_mutation(self, prepared: PreparedMutation) -> PromotionResult:
        def promote(connection: sqlite3.Connection) -> PromotionResult:
            row = connection.execute(
                "SELECT key, generation, locator, expected_lineage, expected_revision, manifest, verified_digest, state FROM mutations WHERE operation_id = ?",
                (prepared.operation_id,),
            ).fetchone()
            if row is None or row[7] != "prepared" or row[6] is None:
                raise CacheBlobLifecycleConflictError("Mutation is not verified and prepared")
            expected = EntryExpectation(row[3], row[4])
            observed = self._expectation(connection, row[0])
            if not self._matches(expected, observed):
                raise CacheBlobLifecycleConflictError("Mutation lineage changed before promotion")
            next_lineage_row = connection.execute(
                "SELECT lineage FROM entry_lineage WHERE key = ?", (row[0],)
            ).fetchone()
            next_lineage = (next_lineage_row[0] if next_lineage_row else 0) + 1
            revision = next_lineage
            connection.execute(
                "INSERT INTO entry_lineage(key, lineage) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET lineage = excluded.lineage",
                (row[0], next_lineage),
            )
            connection.execute(
                "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET generation=excluded.generation, locator=excluded.locator, manifest=excluded.manifest, lineage=excluded.lineage, revision=excluded.revision",
                (row[0], row[1], row[2], row[5], next_lineage, revision),
            )
            connection.execute("UPDATE mutations SET state = 'promoted' WHERE operation_id = ?", (prepared.operation_id,))
            return PromotionResult(EntrySnapshot(row[0], row[1], row[2], bytes(row[5]), EntryExpectation(next_lineage, revision)))
        return self._transaction(promote)

    def abort_mutation(self, prepared: PreparedMutation) -> None:
        self._transaction(lambda connection: connection.execute("DELETE FROM mutations WHERE operation_id = ?", (prepared.operation_id,)))

    def delete_entry(self, key: str, *, expected: EntryExpectation) -> None:
        def delete(connection: sqlite3.Connection) -> None:
            if self._expectation(connection, key) != expected:
                raise CacheBlobLifecycleConflictError("Delete expectation no longer matches authority")
            current = connection.execute("SELECT lineage FROM entry_lineage WHERE key = ?", (key,)).fetchone()
            next_lineage = (current[0] if current else 0) + 1
            connection.execute("DELETE FROM entries WHERE key = ?", (key,))
            connection.execute(
                "INSERT INTO entry_lineage(key, lineage) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET lineage = excluded.lineage",
                (key, next_lineage),
            )
        self._transaction(delete)

    def retire_tombstone(self, key: str, *, expected: EntryExpectation) -> None:
        self.delete_entry(key, expected=expected)

    def begin_clear(self) -> PageToken: return PageToken(uuid4().hex)
    def page_clear(self, token: PageToken) -> tuple[EntrySnapshot, ...]: return ()
    def checkpoint_clear(self, token: PageToken) -> None: return None
    def begin_reconciliation(self) -> PageToken: return PageToken(uuid4().hex)
    def page_reconciliation(self, token: PageToken) -> tuple[CleanupDebt, ...]: return ()
    def checkpoint_reconciliation(self, token: PageToken) -> None: return None
    def compare_and_mark_projection(self, expected: ProjectionRevision | None) -> ProjectionRevision: return ProjectionRevision(0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            self._connection.close()
            self._connection = None


__all__ = ["AUTHORITY_RELATIVE_PATH", "SCHEMA_VERSION", "SQLITE_APPLICATION_ID", "SqliteLifecycleAuthority"]
