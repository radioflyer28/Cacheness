"""SQLite-backed local transactional authority for BlobStore lifecycle state.

The authority stores only bounded lifecycle records. Native handler payloads are
deliberately absent from this module, keeping SQLite writer transactions short.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import tempfile
from threading import Lock
import time
from typing import Callable, Iterator, TypeVar
from uuid import uuid4

from cacheness.config import LifecycleAuthorityTopology, LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobMigrationRequiredError,
    CacheBlobStoreClosedError,
    CacheReason,
)

from .lifecycle_authority import (
    AuthorityCapabilities,
    AuthorityStateSnapshot,
    CleanupDebt,
    EntryExpectation,
    EntrySnapshot,
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


AUTHORITY_RELATIVE_PATH = Path(".cacheness") / "lifecycle-authority-v1.sqlite3"
_BOOTSTRAP_ROOT_NAMES = frozenset(
    {
        ".cacheness",
        "generations",
        "blob_manifest_hmac_key.bin",
        "blob_manifest_hmac_key.bin.ready",
        "blob_manifest_hmac_key.bin.initializing.lock",
    }
)
SQLITE_APPLICATION_ID = 0x43414348
SCHEMA_VERSION = 1
_MAX_STORE_IDENTITY_BYTES = 64
_T = TypeVar("_T")


def _platform_name() -> str:
    """Resolve platform through a narrow contract-test seam."""
    return os.name


class SqliteLifecycleAuthority:
    """One SQLite authority with method-scoped, process-owned connections."""

    capabilities = AuthorityCapabilities(durable=True, multiprocess=True)

    def __init__(
        self,
        root: Path | str,
        *,
        lifecycle_limits: LifecycleLimits | None = None,
        lifecycle_topology: LifecycleAuthorityTopology | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.path = self.root / AUTHORITY_RELATIVE_PATH
        self.lifecycle_limits = (
            LifecycleLimits() if lifecycle_limits is None else lifecycle_limits
        )
        if not isinstance(self.lifecycle_limits, LifecycleLimits):
            raise TypeError("lifecycle_limits must be a LifecycleLimits instance")
        self.lifecycle_topology = (
            LifecycleAuthorityTopology()
            if lifecycle_topology is None
            else lifecycle_topology
        )
        if not isinstance(self.lifecycle_topology, LifecycleAuthorityTopology):
            raise TypeError("lifecycle_topology must be a LifecycleAuthorityTopology instance")
        self._owner_pid = os.getpid()
        self._state_lock = Lock()
        self._closed = False
        self.open_write_transactions = 0
        self._transaction_hook: Callable[[str], None] | None = None

    @classmethod
    def for_root(
        cls,
        root: Path | str,
        *,
        lifecycle_limits: LifecycleLimits | None = None,
        lifecycle_topology: LifecycleAuthorityTopology | None = None,
    ) -> "SqliteLifecycleAuthority":
        """Return a non-materializing authority; mutation creates its database."""
        return cls(
            root,
            lifecycle_limits=lifecycle_limits,
            lifecycle_topology=lifecycle_topology,
        )

    def _require_owned_open(self) -> None:
        if self._closed:
            raise CacheBlobStoreClosedError("Lifecycle authority is closed")
        if os.getpid() != self._owner_pid:
            raise CacheBlobBackendError(
                "Lifecycle authority cannot be reused after fork",
                context={"operation": "lifecycle_authority_process_ownership"},
            )

    def _deadline(self, deadline: float | None) -> float:
        if deadline is not None:
            if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
                raise TypeError("deadline must be a monotonic timestamp")
            return float(deadline)
        return time.monotonic() + self.lifecycle_limits.authority_busy_timeout_seconds

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CacheBlobLifecycleTimeoutError(
                "Lifecycle authority busy deadline expired",
                context={"operation": "lifecycle_authority"},
            )
        return remaining

    def _classify_for_open(self) -> str:
        """Classify authority objects without opening SQLite or creating a path."""
        try:
            root_stat = self.root.lstat()
        except FileNotFoundError:
            return "missing"
        if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
            return "wrong_root"

        reserved = self.root / AUTHORITY_RELATIVE_PATH.parent
        try:
            reserved_stat = reserved.lstat()
        except FileNotFoundError:
            return "ready" if not any(self.root.iterdir()) else "established"
        if not stat.S_ISDIR(reserved_stat.st_mode) or stat.S_ISLNK(reserved_stat.st_mode):
            return "invalid_reserved_directory"

        try:
            database_stat = self.path.lstat()
        except FileNotFoundError:
            return "ready" if not any(reserved.iterdir()) else "established"
        if not stat.S_ISREG(database_stat.st_mode) or stat.S_ISLNK(database_stat.st_mode):
            return "invalid_authority"
        return "authority"

    def _is_pristine_reserved_bootstrap(self) -> bool:
        """Recognize only the empty namespace between root and leaf creation.

        The first mutator creates ``.cacheness`` before it can create the
        exclusive SQLite leaf. Another mutator may observe that narrow window.
        It is safe to join the bootstrap only when the root contains exactly
        that contained, non-symlink directory and the directory is empty;
        every other established root remains a migration-required store.
        """
        reserved = self.root / AUTHORITY_RELATIVE_PATH.parent
        try:
            root_entries = tuple(self.root.iterdir())
            reserved_stat = reserved.lstat()
        except FileNotFoundError:
            return False
        return (
            root_entries == (reserved,)
            and stat.S_ISDIR(reserved_stat.st_mode)
            and not stat.S_ISLNK(reserved_stat.st_mode)
            and not any(reserved.iterdir())
        )

    def _may_be_inflight_authority_bootstrap(self) -> bool:
        """Return whether only new-authority bootstrap artifacts are present."""
        reserved = self.root / AUTHORITY_RELATIVE_PATH.parent
        try:
            root_entries = tuple(self.root.iterdir())
            reserved_stat = reserved.lstat()
        except FileNotFoundError:
            return False
        return (
            self.path not in root_entries
            and all(entry.name in _BOOTSTRAP_ROOT_NAMES for entry in root_entries)
            and reserved in root_entries
            and stat.S_ISDIR(reserved_stat.st_mode)
            and not stat.S_ISLNK(reserved_stat.st_mode)
            and not any(reserved.iterdir())
        )

    def _await_inflight_authority_leaf(self, deadline: float) -> bool:
        """Join a bounded concurrent bootstrap without adopting old evidence."""
        while True:
            if self._classify_for_open() == "authority":
                return False
            if not self._may_be_inflight_authority_bootstrap():
                self._reject_non_authority_state("established")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._reject_non_authority_state("established")
            time.sleep(min(0.001, remaining))

    def _reject_non_authority_state(self, state: str) -> None:
        raise CacheBlobMigrationRequiredError(
            "Store has evidence but no valid lifecycle authority",
            context={"authority_path": str(self.path), "classification": state},
        )

    def _windows_offline_provisioning_command(self) -> str:
        """Return deployment-only guidance without creating or changing the root."""
        root = str(self.root).replace("'", "''")
        return (
            "$root = '"
            + root
            + "'; $logonSid = (whoami /groups /fo csv | ConvertFrom-Csv | "
            "Where-Object { $_.SID -match '^S-1-5-5-[0-9]+-[0-9]+$' } | "
            "Select-Object -First 1 -ExpandProperty SID); "
            "if (-not $logonSid) { throw 'No current-token logon SID found.' }; "
            "icacls.exe $root /inheritance:r; "
            "icacls.exe $root /grant:r \"*$logonSid:(OI)(CI)(M)\" "
            "\"*S-1-5-18:(OI)(CI)(RX)\" \"*S-1-5-32-544:(OI)(CI)(RX)\"; "
            "icacls.exe $root /verify"
        )

    def _raise_windows_topology_error(self, message: str) -> None:
        command = self._windows_offline_provisioning_command()
        raise CacheBlobBackendError(
            f"{message}. Provision the root offline before Cacheness starts: {command}",
            context={
                "authority_path": str(self.path),
                "offline_provisioning": command,
            },
            reason=CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED,
        )

    @staticmethod
    def _run_windows_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Run a read-only Windows inspection command without a shell."""
        return subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
        )

    def _current_windows_logon_sid(self) -> str:
        result = self._run_windows_command(["whoami", "/groups", "/fo", "csv", "/nh"])
        if result.returncode != 0:
            self._raise_windows_topology_error("Current token logon SID cannot be inspected")
        for row in csv.reader(result.stdout.splitlines()):
            for value in row:
                if value.startswith("S-1-5-5-") and value.count("-") == 5:
                    return value
        self._raise_windows_topology_error("Current token has no logon SID")

    def _read_windows_acl_evidence(self) -> dict[str, object]:
        verify = self._run_windows_command(["icacls.exe", str(self.root), "/verify"])
        if verify.returncode != 0:
            self._raise_windows_topology_error("Windows root DACL verification failed")
        script = (
            "$root = $args[0]; $identity = "
            "[System.Security.Principal.WindowsIdentity]::GetCurrent(); "
            "$acl = Get-Acl -LiteralPath $root; "
            "[pscustomobject]@{ account_sid = $identity.User.Value; "
            "protected = $acl.AreAccessRulesProtected; rules = @($acl.Access | "
            "ForEach-Object { [pscustomobject]@{ sid = $_.IdentityReference.Translate("
            "[System.Security.Principal.SecurityIdentifier]).Value; "
            "rights = $_.FileSystemRights.ToString(); type = $_.AccessControlType.ToString(); "
            "inherited = $_.IsInherited } }) } | ConvertTo-Json -Compress"
        )
        result = self._run_windows_command(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script, str(self.root)]
        )
        if result.returncode != 0:
            self._raise_windows_topology_error("Windows root ACL cannot be inspected")
        try:
            evidence = json.loads(result.stdout)
        except json.JSONDecodeError:
            self._raise_windows_topology_error("Windows root ACL proof is malformed")
        if not isinstance(evidence, dict):
            self._raise_windows_topology_error("Windows root ACL proof is malformed")
        return evidence

    @staticmethod
    def _windows_rights_allow_mutation(rights: object) -> bool:
        if not isinstance(rights, str):
            return True
        normalized = rights.replace(" ", "").lower()
        mutation_rights = (
            "fullcontrol",
            "modify",
            "write",
            "appenddata",
            "createfiles",
            "deleted",
            "delete",
            "changepermissions",
            "takeownership",
        )
        return any(right in normalized for right in mutation_rights)

    def _validate_windows_root(self) -> None:
        """Prove the pre-provisioned one-logon-session DACL without modifying it."""
        try:
            root_stat = self.root.lstat()
        except FileNotFoundError:
            self._raise_windows_topology_error("Windows lifecycle root is absent")
        if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
            self._raise_windows_topology_error("Windows lifecycle root is unsafe")
        logon_sid = self._current_windows_logon_sid()
        evidence = self._read_windows_acl_evidence()
        account_sid = evidence.get("account_sid")
        rules = evidence.get("rules")
        if (
            not isinstance(account_sid, str)
            or not account_sid
            or not isinstance(evidence.get("protected"), bool)
            or evidence["protected"] is not True
            or not isinstance(rules, list)
        ):
            self._raise_windows_topology_error("Windows lifecycle root DACL is unprovable")

        allowed_non_mutating = {"S-1-5-18", "S-1-5-32-544"}
        logon_mutation_grant = False
        for rule in rules:
            if not isinstance(rule, dict):
                self._raise_windows_topology_error("Windows lifecycle root DACL is malformed")
            sid = rule.get("sid")
            inherited = rule.get("inherited")
            access_type = rule.get("type")
            mutates = self._windows_rights_allow_mutation(rule.get("rights"))
            if not isinstance(sid, str) or not sid or inherited is not False:
                self._raise_windows_topology_error("Windows lifecycle root DACL has drifted")
            if access_type != "Allow":
                self._raise_windows_topology_error("Windows lifecycle root DACL has unsupported ACEs")
            if sid == logon_sid:
                logon_mutation_grant = logon_mutation_grant or mutates
                continue
            if sid == account_sid or mutates or sid not in allowed_non_mutating:
                self._raise_windows_topology_error("Windows lifecycle root DACL has unsafe grants")
        if not logon_mutation_grant:
            self._raise_windows_topology_error("Windows lifecycle root lacks the logon-SID mutation grant")

    def _validate_mutation_topology(self) -> None:
        """Reject unsupported declared topology before SQLite or payload effects."""
        if self.lifecycle_topology.filesystem != "local" or (
            self.lifecycle_topology.principal_scope != "current_user_current_session"
        ):
            raise CacheBlobBackendError(
                "Lifecycle authority topology is unsupported",
                context={"topology": self.lifecycle_topology.principal_scope},
                reason=CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED,
            )
        if _platform_name() == "nt":
            self._validate_windows_root()

    def _materialize_database_file(self, deadline: float) -> bool:
        """Create only the contained database leaf and report whether this won."""
        self._validate_mutation_topology()
        state = self._classify_for_open()
        rejected_states = {
            "wrong_root",
            "invalid_reserved_directory",
            "invalid_authority",
        }
        if state in rejected_states:
            self._reject_non_authority_state(state)
        if state == "established":
            # The directory/file classification is necessarily a read-only
            # observation. Re-check immediately before rejecting so a sibling
            # that won the O_EXCL leaf creation is never mistaken for legacy
            # evidence in this small TOCTOU window.
            if self._classify_for_open() == "authority":
                return False
            if self._is_pristine_reserved_bootstrap():
                pass
            elif self._may_be_inflight_authority_bootstrap():
                return self._await_inflight_authority_leaf(deadline)
            elif self._classify_for_open() == "authority":
                return False
            else:
                self._reject_non_authority_state(state)
        if state == "missing":
            self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        reserved = self.root / AUTHORITY_RELATIVE_PATH.parent
        if not reserved.exists():
            try:
                reserved.mkdir(mode=0o700, exist_ok=False)
            except FileExistsError:
                # A concurrent first writer may have created the reserved
                # namespace after our classification. Revalidate its exact
                # object type below; never treat this race as success alone.
                pass
        reserved_stat = reserved.lstat()
        if not stat.S_ISDIR(reserved_stat.st_mode) or stat.S_ISLNK(reserved_stat.st_mode):
            self._reject_non_authority_state("invalid_reserved_directory")

        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            database_stat = self.path.lstat()
            if not stat.S_ISREG(database_stat.st_mode) or stat.S_ISLNK(database_stat.st_mode):
                self._reject_non_authority_state("invalid_authority")
            return False
        else:
            os.close(descriptor)
            return True

    @staticmethod
    def _is_busy_error(error: sqlite3.Error) -> bool:
        message = str(error).lower()
        return "busy" in message or "locked" in message

    def _translate_sqlite_error(self, error: sqlite3.Error, *, operation: str) -> None:
        if self._is_busy_error(error):
            raise CacheBlobLifecycleTimeoutError(
                "Lifecycle authority busy deadline expired",
                context={"operation": operation},
            ) from error
        raise CacheBlobBackendError(
            "Lifecycle authority SQLite operation failed",
            context={"operation": operation},
        ) from error

    def set_transaction_hook_for_test(self, hook: Callable[[str], None] | None) -> None:
        """Install one deterministic transition seam for authority fault tests."""
        self._transaction_hook = hook

    def _reach_transaction_boundary(self, boundary: str) -> None:
        if self._transaction_hook is not None:
            self._transaction_hook(boundary)

    @staticmethod
    def _configure_connection(
        connection: sqlite3.Connection,
        *,
        initialize: bool,
    ) -> None:
        """Configure and read back every connection-level authority pragma."""
        if initialize:
            mode = connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
        else:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        connection.execute("PRAGMA synchronous = EXTRA")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")

        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        trusted_schema = connection.execute("PRAGMA trusted_schema").fetchone()[0]
        if (
            str(mode).lower() != "delete"
            or synchronous != 3
            or foreign_keys != 1
            or trusted_schema != 0
        ):
            raise CacheBlobBackendError(
                "Lifecycle authority durability settings are unavailable",
                context={"operation": "lifecycle_authority_configure"},
                reason=CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED,
            )

    @classmethod
    def _initialize_schema(cls, connection: sqlite3.Connection) -> None:
        """Create the complete version-one normalized authority schema once."""
        connection.execute("BEGIN EXCLUSIVE")
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS store_identity (identity TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entry_lineage (key TEXT PRIMARY KEY, lineage INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                "key TEXT PRIMARY KEY, generation TEXT NOT NULL, locator TEXT NOT NULL, "
                "manifest BLOB NOT NULL, manifest_digest TEXT NOT NULL, lineage INTEGER NOT NULL, "
                "revision INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS mutations ("
                "operation_id TEXT PRIMARY KEY, key TEXT NOT NULL, generation TEXT NOT NULL, "
                "locator TEXT NOT NULL, expected_lineage INTEGER, expected_revision INTEGER, "
                "expected_generation TEXT, expected_manifest_digest TEXT, manifest BLOB NOT NULL, "
                "verified_digest TEXT, verified_size INTEGER, state TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS authority_state ("
                "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), revision INTEGER NOT NULL, "
                "projection_dirty INTEGER NOT NULL CHECK (projection_dirty IN (0, 1)))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cleanup_debt ("
                "debt_id INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT NOT NULL, key TEXT NOT NULL, "
                "generation TEXT NOT NULL, locator TEXT NOT NULL, role TEXT NOT NULL, "
                "state TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS clear_runs ("
                "run_id TEXT PRIMARY KEY, state TEXT NOT NULL, revision INTEGER NOT NULL, "
                "last_key TEXT NOT NULL DEFAULT '')"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS clear_targets ("
                "run_id TEXT NOT NULL, key TEXT NOT NULL, lineage INTEGER NOT NULL, "
                "entry_revision INTEGER NOT NULL, generation TEXT NOT NULL, locator TEXT NOT NULL, "
                "manifest BLOB NOT NULL, manifest_digest TEXT NOT NULL, state TEXT NOT NULL, "
                "PRIMARY KEY (run_id, key))"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS clear_targets_page "
                "ON clear_targets(run_id, state, key)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reconciliation_runs ("
                "run_id TEXT PRIMARY KEY, state TEXT NOT NULL, mutation_high_water INTEGER NOT NULL, "
                "debt_high_water INTEGER NOT NULL, authority_revision INTEGER NOT NULL, "
                "mutation_cursor INTEGER NOT NULL DEFAULT 0, debt_cursor INTEGER NOT NULL DEFAULT 0)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reconciliation_actions ("
                "run_id TEXT NOT NULL, action_id INTEGER NOT NULL, state TEXT NOT NULL, "
                "PRIMARY KEY (run_id, action_id))"
            )
            connection.execute(
                "INSERT INTO store_identity(identity) "
                "SELECT lower(hex(randomblob(16))) "
                "WHERE NOT EXISTS (SELECT 1 FROM store_identity)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO authority_state(singleton, revision, projection_dirty) "
                "VALUES (1, 0, 0)"
            )
            connection.execute("PRAGMA application_id = 1128350536")
            connection.execute("PRAGMA user_version = 1")
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @classmethod
    def _migrate_schema_layout(cls, connection: sqlite3.Connection) -> None:
        """Complete the fixed version-one layout before a mutation uses it.

        Plan 03-02's tracer used the same persistent user version but did not
        yet require the corroborating columns or complete transition tables.
        This ordered, one-transaction layout completion preserves the confirmed
        authority identity rather than treating that known predecessor as JSON
        reconstruction input.
        """
        connection.execute("BEGIN EXCLUSIVE")
        try:
            # Read the layout only after taking SQLite's exclusive writer
            # transaction. A concurrent first-use creator can expose a
            # partially initialized schema to a second connector; deciding
            # which ALTER statements to run before this boundary races that
            # creator and can attempt the same migration twice.
            entry_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(entries)")
            }
            mutation_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(mutations)")
            }
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "entries" not in table_names:
                connection.execute("ROLLBACK")
                cls._initialize_schema(connection)
                return

            clear_run_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(clear_runs)")
            } if "clear_runs" in table_names else set()
            clear_target_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(clear_targets)")
            } if "clear_targets" in table_names else set()
            reconciliation_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(reconciliation_runs)")
            } if "reconciliation_runs" in table_names else set()

            needs_migration = (
                "manifest_digest" not in entry_columns
                or "expected_generation" not in mutation_columns
                or "expected_manifest_digest" not in mutation_columns
                or "authority_state" not in table_names
                or "last_key" not in clear_run_columns
                or not {
                    "lineage",
                    "entry_revision",
                    "locator",
                    "manifest",
                }.issubset(clear_target_columns)
                or not {
                    "authority_revision",
                    "mutation_cursor",
                    "debt_cursor",
                }.issubset(reconciliation_columns)
            )
            if not needs_migration:
                connection.execute("COMMIT")
                return
            if "manifest_digest" not in entry_columns:
                connection.execute("ALTER TABLE entries ADD COLUMN manifest_digest TEXT")
                rows = connection.execute("SELECT key, manifest FROM entries").fetchall()
                for key, manifest in rows:
                    if not isinstance(manifest, bytes):
                        raise CacheBlobMigrationRequiredError(
                            "Lifecycle authority entry cannot be migrated safely"
                        )
                    connection.execute(
                        "UPDATE entries SET manifest_digest = ? WHERE key = ?",
                        (hashlib.sha256(manifest).hexdigest(), key),
                    )
            if "expected_generation" not in mutation_columns:
                connection.execute("ALTER TABLE mutations ADD COLUMN expected_generation TEXT")
            if "expected_manifest_digest" not in mutation_columns:
                connection.execute(
                    "ALTER TABLE mutations ADD COLUMN expected_manifest_digest TEXT"
                )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS authority_state ("
                "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), revision INTEGER NOT NULL, "
                "projection_dirty INTEGER NOT NULL CHECK (projection_dirty IN (0, 1)))"
            )
            connection.execute(
                "INSERT OR IGNORE INTO authority_state(singleton, revision, projection_dirty) "
                "VALUES (1, 0, 0)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cleanup_debt ("
                "debt_id INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT NOT NULL, "
                "key TEXT NOT NULL, generation TEXT NOT NULL, locator TEXT NOT NULL, "
                "role TEXT NOT NULL, state TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS clear_runs ("
                "run_id TEXT PRIMARY KEY, state TEXT NOT NULL, revision INTEGER NOT NULL, "
                "last_key TEXT NOT NULL DEFAULT '')"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS clear_targets ("
                "run_id TEXT NOT NULL, key TEXT NOT NULL, lineage INTEGER NOT NULL, "
                "entry_revision INTEGER NOT NULL, generation TEXT NOT NULL, locator TEXT NOT NULL, "
                "manifest BLOB NOT NULL, manifest_digest TEXT NOT NULL, state TEXT NOT NULL, "
                "PRIMARY KEY (run_id, key))"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS clear_targets_page "
                "ON clear_targets(run_id, state, key)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reconciliation_runs ("
                "run_id TEXT PRIMARY KEY, state TEXT NOT NULL, "
                "mutation_high_water INTEGER NOT NULL, debt_high_water INTEGER NOT NULL, "
                "authority_revision INTEGER NOT NULL, mutation_cursor INTEGER NOT NULL DEFAULT 0, "
                "debt_cursor INTEGER NOT NULL DEFAULT 0)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reconciliation_actions ("
                "run_id TEXT NOT NULL, action_id INTEGER NOT NULL, state TEXT NOT NULL, "
                "PRIMARY KEY (run_id, action_id))"
            )
            if "clear_runs" in table_names and "last_key" not in clear_run_columns:
                connection.execute(
                    "ALTER TABLE clear_runs ADD COLUMN last_key TEXT NOT NULL DEFAULT ''"
                )
            for column, declaration in (
                ("lineage", "INTEGER"),
                ("entry_revision", "INTEGER"),
                ("locator", "TEXT"),
                ("manifest", "BLOB"),
            ):
                if (
                    "clear_targets" in table_names
                    and column not in clear_target_columns
                ):
                    connection.execute(
                        f"ALTER TABLE clear_targets ADD COLUMN {column} {declaration}"
                    )
            connection.execute(
                "UPDATE clear_targets SET "
                "lineage = (SELECT lineage FROM entries WHERE entries.key = clear_targets.key), "
                "entry_revision = (SELECT revision FROM entries WHERE entries.key = clear_targets.key), "
                "locator = (SELECT locator FROM entries WHERE entries.key = clear_targets.key), "
                "manifest = (SELECT manifest FROM entries WHERE entries.key = clear_targets.key) "
                "WHERE state = 'pending' AND manifest IS NULL"
            )
            connection.execute(
                "UPDATE clear_targets SET state = 'conflicted' "
                "WHERE state = 'pending' AND (lineage IS NULL OR entry_revision IS NULL "
                "OR locator IS NULL OR manifest IS NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS clear_targets_page "
                "ON clear_targets(run_id, state, key)"
            )
            if "reconciliation_runs" in table_names:
                for column, declaration in (
                    ("authority_revision", "INTEGER NOT NULL DEFAULT 0"),
                    ("mutation_cursor", "INTEGER NOT NULL DEFAULT 0"),
                    ("debt_cursor", "INTEGER NOT NULL DEFAULT 0"),
                ):
                    if column not in reconciliation_columns:
                        connection.execute(
                            f"ALTER TABLE reconciliation_runs ADD COLUMN {column} {declaration}"
                        )
                connection.execute(
                    "UPDATE reconciliation_runs SET authority_revision = "
                    "(SELECT revision FROM authority_state WHERE singleton = 1) "
                    "WHERE authority_revision = 0"
                )
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @classmethod
    def _validate_schema(cls, connection: sqlite3.Connection) -> None:
        """Reject an unknown database before it participates in a transition."""
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if application_id != SQLITE_APPLICATION_ID:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority application ID is incompatible"
            )
        if version > SCHEMA_VERSION:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority schema version is unsupported"
            )
        if version != SCHEMA_VERSION:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority schema version is incompatible"
            )

        rows = connection.execute("SELECT identity FROM store_identity").fetchall()
        if len(rows) != 1 or not isinstance(rows[0][0], str):
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority store identity is incompatible"
            )
        identity = rows[0][0]
        if not identity or len(identity.encode("utf-8")) > _MAX_STORE_IDENTITY_BYTES:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority store identity is incompatible"
            )

    @contextmanager
    def _connection(
        self,
        *,
        mutation: bool,
        deadline: float,
    ) -> Iterator[sqlite3.Connection | None]:
        """Open, harden, validate, and close one process/thread-owned connection."""
        self._require_owned_open()
        created_new = False
        if mutation:
            self._remaining(deadline)
            created_new = self._materialize_database_file(deadline)
        else:
            state = self._classify_for_open()
            if state == "established" and self._is_pristine_reserved_bootstrap():
                state = "ready"
            if state in {"missing", "ready"}:
                yield None
                return
            if state != "authority":
                self._reject_non_authority_state(state)

        connection: sqlite3.Connection | None = None
        try:
            remaining = self._remaining(deadline)
            if mutation:
                connection = sqlite3.connect(
                    self.path,
                    isolation_level=None,
                    timeout=remaining,
                    check_same_thread=True,
                )
            else:
                connection = sqlite3.connect(
                    f"{self.path.as_uri()}?mode=ro",
                    uri=True,
                    isolation_level=None,
                    timeout=remaining,
                    check_same_thread=True,
                )
            self._configure_connection(connection, initialize=created_new)
            if created_new:
                self._initialize_schema(connection)
            elif mutation:
                self._migrate_schema_layout(connection)
            application_id = connection.execute("PRAGMA application_id").fetchone()[0]
            table_rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            if application_id == 0 and not table_rows:
                # An O_EXCL winner made the database leaf visible but has not
                # committed its first schema transaction yet. SQLite remains
                # the only cross-process authority: close this observer and
                # retry the exact bounded open rather than classifying a live
                # zero-byte bootstrap as incompatible stored evidence.
                connection.close()
                connection = None
                time.sleep(
                    min(
                        0.001,
                        self._remaining(deadline),
                    )
                )
                with self._connection(mutation=mutation, deadline=deadline) as retry:
                    yield retry
                return
            self._validate_schema(connection)
            yield connection
        except (
            CacheBlobBackendError,
            CacheBlobLifecycleTimeoutError,
            CacheBlobMigrationRequiredError,
        ):
            raise
        except sqlite3.Error as error:
            self._translate_sqlite_error(error, operation="lifecycle_authority_open")
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _expectation(connection: sqlite3.Connection, key: str) -> EntryExpectation:
        row = connection.execute(
            "SELECT lineage, revision, generation, manifest_digest FROM entries WHERE key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            return EntryExpectation(
                lineage=row[0],
                revision=row[1],
                generation=row[2],
                manifest_digest=row[3],
            )
        row = connection.execute(
            "SELECT lineage FROM entry_lineage WHERE key = ?", (key,)
        ).fetchone()
        return EntryExpectation(lineage=None if row is None else row[0], revision=None)

    @staticmethod
    def _matches(expected: EntryExpectation, observed: EntryExpectation) -> bool:
        return expected == observed

    def _transaction(
        self,
        callback: Callable[[sqlite3.Connection], _T],
        *,
        deadline: float | None = None,
        uncertain_classifier: Callable[[], _T] | None = None,
    ) -> _T:
        """Run one bounded state transition with rollback on every failure."""
        absolute_deadline = self._deadline(deadline)
        with self._connection(mutation=True, deadline=absolute_deadline) as connection:
            assert connection is not None
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as error:
                self._translate_sqlite_error(error, operation="lifecycle_authority_begin")
            with self._state_lock:
                self.open_write_transactions += 1
            try:
                result = callback(connection)
                self._reach_transaction_boundary("authority.transaction.before_commit")
                connection.execute("COMMIT")
                self._reach_transaction_boundary("authority.transaction.committed")
                return result
            except BaseException as error:
                committed = not connection.in_transaction
                if connection.in_transaction:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                if (
                    committed
                    and uncertain_classifier is not None
                    and isinstance(error, sqlite3.Error)
                ):
                    return uncertain_classifier()
                raise
            finally:
                with self._state_lock:
                    self.open_write_transactions -= 1

    def read_entry(self, key: str) -> EntrySnapshot | None:
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                return None
            try:
                row = connection.execute(
                    "SELECT generation, locator, manifest, manifest_digest, lineage, revision "
                    "FROM entries WHERE key = ?",
                    (key,),
                ).fetchone()
            except sqlite3.Error as error:
                self._translate_sqlite_error(error, operation="lifecycle_authority_read")
            if row is None:
                return None
            try:
                manifest = bytes(row[2])
                if hashlib.sha256(manifest).hexdigest() != row[3]:
                    raise ValueError("manifest digest does not corroborate the entry")
                return EntrySnapshot(
                    key,
                    row[0],
                    row[1],
                    manifest,
                    EntryExpectation(row[4], row[5], row[0], row[3]),
                )
            except (TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "Lifecycle authority entry row is malformed",
                    context={"operation": "lifecycle_authority_read"},
                ) from error

    def prepare_mutation(self, spec: MutationSpec) -> PreparedMutation:
        def prepare(connection: sqlite3.Connection) -> PreparedMutation:
            existing = connection.execute(
                "SELECT key, generation, locator, expected_lineage, expected_revision, "
                "expected_generation, expected_manifest_digest, manifest "
                "FROM mutations WHERE operation_id = ?",
                (spec.operation_id,),
            ).fetchone()
            expected_values = (
                spec.key,
                spec.generation,
                spec.candidate_locator,
                spec.expected.lineage,
                spec.expected.revision,
                spec.expected.generation,
                spec.expected.manifest_digest,
                spec.manifest,
            )
            if existing is not None:
                if tuple(existing) == expected_values:
                    return PreparedMutation(spec.operation_id, spec)
                raise CacheBlobLifecycleConflictError("Operation identifier is not reusable")
            observed = self._expectation(connection, spec.key)
            if not self._matches(spec.expected, observed):
                raise CacheBlobLifecycleConflictError(
                    "Mutation expectation no longer matches authority"
                )
            connection.execute(
                "INSERT INTO mutations "
                "(operation_id, key, generation, locator, expected_lineage, expected_revision, "
                "expected_generation, expected_manifest_digest, manifest, verified_digest, "
                "verified_size, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 'prepared')",
                (
                    spec.operation_id,
                    spec.key,
                    spec.generation,
                    spec.candidate_locator,
                    spec.expected.lineage,
                    spec.expected.revision,
                    spec.expected.generation,
                    spec.expected.manifest_digest,
                    spec.manifest,
                ),
            )
            return PreparedMutation(spec.operation_id, spec)

        return self._transaction(prepare)

    def record_verification(
        self, prepared: PreparedMutation, proof: VerificationProof
    ) -> None:
        def record(connection: sqlite3.Connection) -> None:
            cursor = connection.execute(
                "UPDATE mutations SET verified_digest = ?, verified_size = ?, manifest = ? "
                "WHERE operation_id = ? AND state = 'prepared'",
                (
                    proof.digest,
                    proof.byte_size,
                    proof.manifest or prepared.spec.manifest,
                    prepared.operation_id,
                ),
            )
            if cursor.rowcount != 1:
                raise CacheBlobLifecycleConflictError(
                    "Prepared mutation cannot accept verification"
                )

        self._transaction(record)

    def record_verification_for_test(self, prepared: PreparedMutation) -> None:
        """Install deterministic proof for authority-only contract tests."""
        self.record_verification(prepared, VerificationProof("0" * 64, 0))

    def promote_mutation(self, prepared: PreparedMutation) -> PromotionResult:
        def promote(connection: sqlite3.Connection) -> PromotionResult:
            row = connection.execute(
                "SELECT key, generation, locator, expected_lineage, expected_revision, "
                "expected_generation, expected_manifest_digest, manifest, verified_digest, state "
                "FROM mutations WHERE operation_id = ?",
                (prepared.operation_id,),
            ).fetchone()
            if row is None:
                raise CacheBlobLifecycleConflictError("Mutation does not exist")
            if row[9] == "promoted":
                return self._promoted_result(connection, prepared.operation_id)
            if row[9] != "prepared" or row[8] is None:
                raise CacheBlobLifecycleConflictError(
                    "Mutation is not verified and prepared"
                )
            expected = EntryExpectation(row[3], row[4], row[5], row[6])
            observed = self._expectation(connection, row[0])
            if not self._matches(expected, observed):
                raise CacheBlobLifecycleConflictError(
                    "Mutation lineage changed before promotion"
                )
            old_entry = connection.execute(
                "SELECT generation, locator FROM entries WHERE key = ?", (row[0],)
            ).fetchone()
            self._reach_transaction_boundary("promote.before_lineage")
            next_lineage_row = connection.execute(
                "SELECT lineage FROM entry_lineage WHERE key = ?", (row[0],)
            ).fetchone()
            next_lineage = (next_lineage_row[0] if next_lineage_row else 0) + 1
            revision = connection.execute(
                "SELECT revision FROM authority_state WHERE singleton = 1"
            ).fetchone()[0] + 1
            connection.execute(
                "INSERT INTO entry_lineage(key, lineage) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET lineage = excluded.lineage",
                (row[0], next_lineage),
            )
            self._reach_transaction_boundary("promote.after_lineage")
            manifest_digest = hashlib.sha256(bytes(row[7])).hexdigest()
            connection.execute(
                "INSERT INTO entries(key, generation, locator, manifest, manifest_digest, lineage, revision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                "generation=excluded.generation, locator=excluded.locator, "
                "manifest=excluded.manifest, manifest_digest=excluded.manifest_digest, "
                "lineage=excluded.lineage, "
                "revision=excluded.revision",
                (row[0], row[1], row[2], row[7], manifest_digest, next_lineage, revision),
            )
            self._reach_transaction_boundary("promote.after_entry")
            connection.execute(
                "UPDATE mutations SET state = 'promoted' WHERE operation_id = ?",
                (prepared.operation_id,),
            )
            self._reach_transaction_boundary("promote.after_mutation")
            if old_entry is not None and old_entry[1] != row[2]:
                connection.execute(
                    "INSERT INTO cleanup_debt(operation_id, key, generation, locator, role, state) "
                    "VALUES (?, ?, ?, ?, 'previous_generation', 'pending')",
                    (prepared.operation_id, row[0], old_entry[0], old_entry[1]),
                )
            self._reach_transaction_boundary("promote.after_cleanup_debt")
            connection.execute(
                "UPDATE authority_state SET revision = ?, projection_dirty = 1 WHERE singleton = 1",
                (revision,),
            )
            self._reach_transaction_boundary("promote.after_projection")
            return self._promoted_result(connection, prepared.operation_id)

        return self._transaction(
            promote,
            uncertain_classifier=lambda: self._classify_promoted_mutation(prepared),
        )

    def _promoted_result(
        self, connection: sqlite3.Connection, operation_id: str
    ) -> PromotionResult:
        """Read one already-committed promotion without reapplying it."""
        row = connection.execute(
            "SELECT m.key, e.generation, e.locator, e.manifest, e.manifest_digest, "
            "e.lineage, e.revision FROM mutations AS m JOIN entries AS e ON e.key = m.key "
            "WHERE m.operation_id = ? AND m.state = 'promoted'",
            (operation_id,),
        ).fetchone()
        if row is None or hashlib.sha256(bytes(row[3])).hexdigest() != row[4]:
            raise CacheBlobBackendError(
                "Committed lifecycle authority promotion is malformed",
                context={"operation": "lifecycle_authority_promote"},
            )
        debt_rows = connection.execute(
            "SELECT operation_id, locator, key, generation, role FROM cleanup_debt "
            "WHERE operation_id = ? AND state = 'pending' ORDER BY debt_id",
            (operation_id,),
        ).fetchall()
        return PromotionResult(
            EntrySnapshot(
                row[0],
                row[1],
                row[2],
                bytes(row[3]),
                EntryExpectation(row[5], row[6], row[1], row[4]),
            ),
            tuple(CleanupDebt(*debt_row) for debt_row in debt_rows),
        )

    def _classify_promoted_mutation(self, prepared: PreparedMutation) -> PromotionResult:
        """Resolve an ambiguous commit by reading exact durable operation state."""
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                raise CacheBlobBackendError(
                    "Lifecycle authority disappeared while classifying commit uncertainty",
                    context={"operation": "lifecycle_authority_promote"},
                )
            return self._promoted_result(connection, prepared.operation_id)

    def abort_mutation(
        self, prepared: PreparedMutation, *, candidate_persisted: bool = False
    ) -> None:
        """Retire pre-publication intent or retain exact candidate cleanup debt."""
        def abort(connection: sqlite3.Connection) -> None:
            row = connection.execute(
                "SELECT key, generation, locator, state FROM mutations WHERE operation_id = ?",
                (prepared.operation_id,),
            ).fetchone()
            if row is None or row[3] == "promoted":
                return
            if not candidate_persisted:
                connection.execute(
                    "DELETE FROM mutations WHERE operation_id = ?",
                    (prepared.operation_id,),
                )
                return
            connection.execute(
                "UPDATE mutations SET state = 'aborted' WHERE operation_id = ?",
                (prepared.operation_id,),
            )
            existing = connection.execute(
                "SELECT 1 FROM cleanup_debt WHERE operation_id = ? AND locator = ? "
                "AND role = 'candidate' AND state = 'pending'",
                (prepared.operation_id, row[2]),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO cleanup_debt(operation_id, key, generation, locator, role, state) "
                    "VALUES (?, ?, ?, ?, 'candidate', 'pending')",
                    (prepared.operation_id, row[0], row[1], row[2]),
                )

        self._transaction(abort)

    def list_entries(self) -> tuple[EntrySnapshot, ...]:
        """Return bounded committed/tombstone snapshots through the authority."""
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                return ()
            rows = connection.execute(
                "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision "
                "FROM entries ORDER BY key"
            ).fetchall()
            return tuple(
                EntrySnapshot(
                    row[0],
                    row[1],
                    row[2],
                    bytes(row[3]),
                    EntryExpectation(row[5], row[6], row[1], row[4]),
                )
                for row in rows
            )

    def pending_cleanup_debts(
        self,
        *,
        key: str | None = None,
        operation_id: str | None = None,
    ) -> tuple[CleanupDebt, ...]:
        """Return exact pending debt without turning a projection into authority."""
        filters = ["state = 'pending'"]
        values: list[str] = []
        if key is not None:
            filters.append("key = ?")
            values.append(key)
        if operation_id is not None:
            filters.append("operation_id = ?")
            values.append(operation_id)
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                return ()
            rows = connection.execute(
                "SELECT operation_id, locator, key, generation, role FROM cleanup_debt "
                f"WHERE {' AND '.join(filters)} ORDER BY debt_id",
                values,
            ).fetchall()
            return tuple(CleanupDebt(*row) for row in rows)

    def retire_cleanup_debt(self, debt: CleanupDebt) -> None:
        """Retire one exact, already reclaimed debt idempotently."""
        def retire(connection: sqlite3.Connection) -> None:
            connection.execute(
                "DELETE FROM cleanup_debt WHERE operation_id = ? AND locator = ? "
                "AND key = ? AND generation = ? AND role = ? AND state = 'pending'",
                (
                    debt.operation_id,
                    debt.locator,
                    debt.key,
                    debt.generation,
                    debt.role,
                ),
            )

        self._transaction(retire)

    def pending_mutations(self) -> tuple[PreparedMutation, ...]:
        """Return only indexed pre-promotion operations for recovery."""
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                return ()
            rows = connection.execute(
                "SELECT operation_id, key, generation, locator, expected_lineage, "
                "expected_revision, expected_generation, expected_manifest_digest, manifest "
                "FROM mutations WHERE state = 'prepared' ORDER BY rowid"
            ).fetchall()
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

    def delete_entry(self, key: str, *, expected: EntryExpectation) -> None:
        def delete(connection: sqlite3.Connection) -> None:
            if self._expectation(connection, key) != expected:
                raise CacheBlobLifecycleConflictError(
                    "Delete expectation no longer matches authority"
                )
            current = connection.execute(
                "SELECT lineage FROM entry_lineage WHERE key = ?", (key,)
            ).fetchone()
            next_lineage = (current[0] if current else 0) + 1
            connection.execute("DELETE FROM entries WHERE key = ?", (key,))
            connection.execute(
                "INSERT INTO entry_lineage(key, lineage) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET lineage = excluded.lineage",
                (key, next_lineage),
            )
            revision = connection.execute(
                "SELECT revision FROM authority_state WHERE singleton = 1"
            ).fetchone()[0] + 1
            connection.execute(
                "UPDATE authority_state SET revision = ?, projection_dirty = 1 "
                "WHERE singleton = 1",
                (revision,),
            )

        self._transaction(delete)

    def retire_tombstone(self, key: str, *, expected: EntryExpectation) -> None:
        self.delete_entry(key, expected=expected)

    def begin_clear(self) -> PageToken:
        def begin(connection: sqlite3.Connection) -> PageToken:
            active = connection.execute(
                "SELECT run_id FROM clear_runs WHERE state = 'active' ORDER BY rowid LIMIT 1"
            ).fetchone()
            if active is not None:
                return PageToken(str(active[0]))
            token = PageToken(uuid4().hex)
            revision = connection.execute(
                "SELECT revision FROM authority_state WHERE singleton = 1"
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO clear_runs(run_id, state, revision, last_key) "
                "VALUES (?, 'active', ?, '')",
                (token.value, revision),
            )
            # SQLite owns membership in this one short transaction.  The
            # target rows retain the exact authenticated bytes and expectation
            # needed to revalidate later destructive work without a second
            # inventory or any filename discovery.
            connection.execute(
                "INSERT INTO clear_targets("
                "run_id, key, lineage, entry_revision, generation, locator, manifest, "
                "manifest_digest, state) "
                "SELECT ?, key, lineage, revision, generation, locator, manifest, "
                "manifest_digest, 'pending' FROM entries",
                (token.value,),
            )
            return token

        return self._transaction(begin)

    def page_clear(self, token: PageToken) -> tuple[EntrySnapshot, ...]:
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                return ()
            run = connection.execute(
                "SELECT state, last_key FROM clear_runs WHERE run_id = ?", (token.value,)
            ).fetchone()
            if run is None:
                raise CacheBlobLifecycleConflictError("Clear run does not exist")
            if run[0] != "active":
                return ()
            rows = connection.execute(
                "SELECT key, generation, locator, manifest, manifest_digest, lineage, entry_revision "
                "FROM clear_targets WHERE run_id = ? AND state = 'pending' AND key > ? "
                "ORDER BY key LIMIT ?",
                (token.value, run[1], self.lifecycle_limits.manifest_page_size),
            ).fetchall()
            return tuple(
                EntrySnapshot(
                    row[0],
                    row[1],
                    row[2],
                    bytes(row[3]),
                    EntryExpectation(row[5], row[6], row[1], row[4]),
                )
                for row in rows
            )

    def checkpoint_clear(
        self,
        token: PageToken,
        target: EntrySnapshot | None = None,
        *,
        state: str = "completed",
    ) -> None:
        def checkpoint(connection: sqlite3.Connection) -> None:
            run = connection.execute(
                "SELECT state FROM clear_runs WHERE run_id = ?", (token.value,)
            ).fetchone()
            if target is None and run is not None and run[0] == "completed":
                return
            if run is None or run[0] != "active":
                raise CacheBlobLifecycleConflictError("Clear run cannot accept checkpoint")
            if target is None:
                connection.execute(
                    "UPDATE clear_runs SET state = 'completed' WHERE run_id = ?",
                    (token.value,),
                )
                return
            if state not in {"completed", "conflicted", "blocked"}:
                raise ValueError("Clear target state is unsupported")
            row = connection.execute(
                "SELECT state FROM clear_targets WHERE run_id = ? AND key = ? "
                "AND lineage = ? AND entry_revision = ? AND generation = ? "
                "AND manifest_digest = ?",
                (
                    token.value,
                    target.key,
                    target.expectation.lineage,
                    target.expectation.revision,
                    target.generation,
                    target.expectation.manifest_digest,
                ),
            ).fetchone()
            if row is None or row[0] != "pending":
                raise CacheBlobLifecycleConflictError("Clear target is no longer pending")
            current = connection.execute(
                "SELECT lineage, revision, generation, manifest_digest FROM entries WHERE key = ?",
                (target.key,),
            ).fetchone()
            if state == "completed":
                lineage = connection.execute(
                    "SELECT lineage FROM entry_lineage WHERE key = ?", (target.key,)
                ).fetchone()
                if current is not None or lineage is None or lineage[0] <= target.expectation.lineage:
                    raise CacheBlobLifecycleConflictError(
                        "Clear target completion lacks exact absence proof"
                    )
            elif state == "conflicted" and current is not None and current == (
                target.expectation.lineage,
                target.expectation.revision,
                target.generation,
                target.expectation.manifest_digest,
            ):
                raise CacheBlobLifecycleConflictError("Clear target has not changed")
            cursor = connection.execute(
                "UPDATE clear_targets SET state = ? WHERE run_id = ? AND key = ? "
                "AND state = 'pending'",
                (state, token.value, target.key),
            )
            if cursor.rowcount != 1:
                raise CacheBlobLifecycleConflictError("Clear target cannot accept checkpoint")
            connection.execute(
                "UPDATE clear_runs SET last_key = ?, state = CASE WHEN NOT EXISTS "
                "(SELECT 1 FROM clear_targets WHERE run_id = ? AND state = 'pending') "
                "THEN 'completed' ELSE 'active' END WHERE run_id = ?",
                (target.key, token.value, token.value),
            )

        self._transaction(checkpoint)

    def begin_reconciliation(self) -> PageToken:
        def begin(connection: sqlite3.Connection) -> PageToken:
            active = connection.execute(
                "SELECT run_id FROM reconciliation_runs WHERE state = 'active' "
                "ORDER BY rowid LIMIT 1"
            ).fetchone()
            if active is not None:
                return PageToken(str(active[0]))
            token = PageToken(uuid4().hex)
            mutation_high_water = connection.execute(
                "SELECT COALESCE(MAX(rowid), 0) FROM mutations"
            ).fetchone()[0]
            debt_high_water = connection.execute(
                "SELECT COALESCE(MAX(debt_id), 0) FROM cleanup_debt"
            ).fetchone()[0]
            authority_revision = connection.execute(
                "SELECT revision FROM authority_state WHERE singleton = 1"
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO reconciliation_runs("
                "run_id, state, mutation_high_water, debt_high_water, authority_revision, "
                "mutation_cursor, debt_cursor) VALUES (?, 'active', ?, ?, ?, 0, 0)",
                (token.value, mutation_high_water, debt_high_water, authority_revision),
            )
            return token

        return self._transaction(begin)

    def reconciliation_snapshot(
        self, token: PageToken | None = None
    ) -> ReconciliationSnapshot:
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                return ReconciliationSnapshot(0, 0, 0)
            if token is not None:
                row = connection.execute(
                    "SELECT authority_revision, mutation_high_water, debt_high_water "
                    "FROM reconciliation_runs WHERE run_id = ?",
                    (token.value,),
                ).fetchone()
                if row is None:
                    raise CacheBlobLifecycleConflictError("Reconciliation run does not exist")
                return ReconciliationSnapshot(row[0], row[1], row[2], token.value)
            revision = connection.execute(
                "SELECT revision FROM authority_state WHERE singleton = 1"
            ).fetchone()[0]
            mutation_high_water = connection.execute(
                "SELECT COALESCE(MAX(rowid), 0) FROM mutations"
            ).fetchone()[0]
            debt_high_water = connection.execute(
                "SELECT COALESCE(MAX(debt_id), 0) FROM cleanup_debt"
            ).fetchone()[0]
            return ReconciliationSnapshot(revision, mutation_high_water, debt_high_water)

    def page_reconciliation_work(
        self,
        snapshot: ReconciliationSnapshot,
        *,
        mutation_cursor: int,
        debt_cursor: int,
    ) -> ReconciliationPage:
        if mutation_cursor < 0 or debt_cursor < 0:
            raise ValueError("Reconciliation cursors must be non-negative")
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                return ReconciliationPage((), mutation_cursor, debt_cursor)
            page_size = max(1, self.lifecycle_limits.operation_page_size // 2)
            mutation_rows = connection.execute(
                "SELECT rowid, operation_id, key, generation, locator, expected_lineage, "
                "expected_revision, expected_generation, expected_manifest_digest, manifest, state "
                "FROM mutations WHERE rowid > ? AND rowid <= ? AND state = 'prepared' "
                "ORDER BY rowid LIMIT ?",
                (mutation_cursor, snapshot.mutation_high_water, page_size),
            ).fetchall()
            debt_rows = connection.execute(
                "SELECT debt_id, operation_id, locator, key, generation, role, state "
                "FROM cleanup_debt WHERE debt_id > ? AND debt_id <= ? "
                "ORDER BY debt_id LIMIT ?",
                (debt_cursor, snapshot.debt_high_water, page_size),
            ).fetchall()
            works: list[ReconciliationWork] = []
            for row in mutation_rows:
                prepared = PreparedMutation(
                    row[1],
                    MutationSpec.create(
                        operation_id=row[1],
                        key=row[2],
                        generation=row[3],
                        candidate_locator=row[4],
                        expected=EntryExpectation(row[5], row[6], row[7], row[8]),
                        manifest=bytes(row[9]),
                    ),
                )
                works.append(ReconciliationWork("mutation", row[0], row[10], mutation=prepared))
            for row in debt_rows:
                debt = CleanupDebt(row[1], row[2], row[3], row[4], row[5], row[0])
                works.append(ReconciliationWork("debt", row[0], row[6], debt=debt))
            next_mutation = (
                mutation_rows[-1][0]
                if mutation_rows
                else snapshot.mutation_high_water
            )
            next_debt = debt_rows[-1][0] if debt_rows else snapshot.debt_high_water
            return ReconciliationPage(tuple(works), next_mutation, next_debt)

    def page_reconciliation(self, token: PageToken) -> tuple[CleanupDebt, ...]:
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                return ()
            run = connection.execute(
                "SELECT debt_high_water FROM reconciliation_runs WHERE run_id = ?",
                (token.value,),
            ).fetchone()
            if run is None:
                raise CacheBlobLifecycleConflictError("Reconciliation run does not exist")
            rows = connection.execute(
                "SELECT operation_id, locator, key, generation, role FROM cleanup_debt "
                "WHERE state = 'pending' AND debt_id <= ? ORDER BY debt_id LIMIT ?",
                (run[0], self.lifecycle_limits.operation_page_size),
            ).fetchall()
            return tuple(CleanupDebt(*row) for row in rows)

    def checkpoint_reconciliation(
        self,
        token: PageToken,
        work: ReconciliationWork | None = None,
        *,
        state: str = "completed",
    ) -> None:
        def checkpoint(connection: sqlite3.Connection) -> None:
            if work is not None:
                if state not in {"completed", "blocked", "conflicted"}:
                    raise ValueError("Reconciliation checkpoint state is unsupported")
                column = "mutation_cursor" if work.source == "mutation" else "debt_cursor"
                cursor = connection.execute(
                    f"UPDATE reconciliation_runs SET {column} = MAX({column}, ?) "
                    "WHERE run_id = ? AND state = 'active'",
                    (work.row_id, token.value),
                )
                if cursor.rowcount != 1:
                    raise CacheBlobLifecycleConflictError(
                        "Reconciliation run cannot accept checkpoint"
                    )
                return
            cursor = connection.execute(
                "UPDATE reconciliation_runs SET state = 'completed' "
                "WHERE run_id = ? AND state = 'active'",
                (token.value,),
            )
            if cursor.rowcount != 1:
                raise CacheBlobLifecycleConflictError(
                    "Reconciliation run cannot accept checkpoint"
                )

        self._transaction(checkpoint)

    def compare_and_mark_projection(
        self, expected: ProjectionRevision | None
    ) -> ProjectionRevision:
        def mark(connection: sqlite3.Connection) -> ProjectionRevision:
            revision, dirty = connection.execute(
                "SELECT revision, projection_dirty FROM authority_state WHERE singleton = 1"
            ).fetchone()
            if expected is not None and expected.value != revision:
                raise CacheBlobLifecycleConflictError("Projection revision changed")
            if dirty:
                connection.execute(
                    "UPDATE authority_state SET projection_dirty = 0 WHERE singleton = 1"
                )
            return ProjectionRevision(revision)

        return self._transaction(mark)

    @contextmanager
    def projection_backup(self) -> Iterator[ProjectionBackup]:
        """Yield a private consistent backup without retaining a live source handle.

        Projection consumers must never render while holding an authority
        transaction or connection.  SQLite's online backup copies one
        consistent read snapshot into a mode-restricted temporary database;
        the source connection is closed before the caller receives it.
        """
        absolute_deadline = self._deadline(None)
        snapshot_path: Path | None = None
        try:
            with self._connection(mutation=False, deadline=absolute_deadline) as source:
                if source is None:
                    raise CacheBlobMigrationRequiredError(
                        "Lifecycle authority is absent for projection export"
                    )
                descriptor, name = tempfile.mkstemp(
                    prefix=".lifecycle-projection-",
                    suffix=".sqlite3",
                    dir=self.path.parent,
                )
                os.close(descriptor)
                snapshot_path = Path(name)
                destination = sqlite3.connect(snapshot_path, isolation_level=None)
                try:
                    source.backup(destination)
                    revision = destination.execute(
                        "SELECT revision FROM authority_state WHERE singleton = 1"
                    ).fetchone()
                    if revision is None or type(revision[0]) is not int:
                        raise CacheBlobMigrationRequiredError(
                            "Lifecycle authority projection revision is incompatible"
                        )
                finally:
                    destination.close()
            self._reach_transaction_boundary("projection.backup.closed_source")
            yield ProjectionBackup(snapshot_path, ProjectionRevision(revision[0]))
        except sqlite3.Error as error:
            self._translate_sqlite_error(error, operation="lifecycle_projection_backup")
        finally:
            if snapshot_path is not None:
                try:
                    snapshot_path.unlink()
                except FileNotFoundError:
                    pass

    def snapshot_state(self) -> AuthorityStateSnapshot:
        """Return one bounded authority-state diagnostic without exposing tables."""
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                return AuthorityStateSnapshot(0, False, (), ())
            revision, dirty = connection.execute(
                "SELECT revision, projection_dirty FROM authority_state WHERE singleton = 1"
            ).fetchone()
            mutations = tuple(
                (row[0], row[1])
                for row in connection.execute(
                    "SELECT operation_id, state FROM mutations ORDER BY operation_id"
                )
            )
            debts = tuple(
                CleanupDebt(*row)
                for row in connection.execute(
                    "SELECT operation_id, locator, key, generation, role FROM cleanup_debt "
                    "WHERE state = 'pending' ORDER BY debt_id"
                )
            )
            return AuthorityStateSnapshot(revision, dirty == 1, mutations, debts)

    def diagnostics(self) -> dict[str, object]:
        """Return read-only integrity diagnostics; this method never repairs."""
        absolute_deadline = self._deadline(None)
        with self._connection(mutation=False, deadline=absolute_deadline) as connection:
            if connection is None:
                raise CacheBlobMigrationRequiredError("Lifecycle authority is absent")
            try:
                identity = connection.execute(
                    "SELECT identity FROM store_identity"
                ).fetchone()[0]
                return {
                    "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0],
                    "synchronous": "extra"
                    if connection.execute("PRAGMA synchronous").fetchone()[0] == 3
                    else "unsupported",
                    "foreign_keys": connection.execute("PRAGMA foreign_keys").fetchone()[0]
                    == 1,
                    "trusted_schema": connection.execute("PRAGMA trusted_schema").fetchone()[0]
                    == 1,
                    "application_id": connection.execute("PRAGMA application_id").fetchone()[0],
                    "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
                    "store_identity": identity,
                    "integrity_check": tuple(
                        row[0] for row in connection.execute("PRAGMA integrity_check")
                    ),
                    "foreign_key_check": tuple(
                        row for row in connection.execute("PRAGMA foreign_key_check")
                    ),
                }
            except sqlite3.Error as error:
                self._translate_sqlite_error(error, operation="lifecycle_authority_diagnostics")

    def close(self) -> None:
        """Close the instance boundary; method-scoped SQLite connections are gone."""
        with self._state_lock:
            self._closed = True


__all__ = [
    "AUTHORITY_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "SQLITE_APPLICATION_ID",
    "SqliteLifecycleAuthority",
]
