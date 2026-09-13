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
from typing import Any, Callable, Iterator, TypeVar
from uuid import uuid4

from cacheness.config import LifecycleAuthorityTopology, LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobMigrationOfflineDecisionRequiredError,
    CacheBlobMigrationRequiredError,
    CacheBlobStoreClosedError,
    CacheManifestIntegrityError,
    CacheReason,
)

from .lifecycle_authority import (
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
from .migration_authority import (
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
from .manifest import BlobManifest
from .catalog import (
    CatalogCursor,
    CatalogCursorError,
    CatalogPage,
    CatalogQuery,
    CatalogSchema,
    page_from_canonical_scan,
    validate_catalog_page_request,
)


AUTHORITY_RELATIVE_PATH = Path(".cacheness") / "lifecycle-authority-v2.sqlite3"
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
# SQLite's user_version is a database-schema identifier, not the public store
# format version. Keep it independent so descriptor and payload formats can
# evolve without implying an implicit SQLite migration.
SQLITE_USER_VERSION = 9
# Retained as an import-compatible name for authority diagnostics. It denotes
# the current SQLite schema only; it is deliberately not STORE_FORMAT_VERSION.
SCHEMA_VERSION = SQLITE_USER_VERSION
_MAX_STORE_IDENTITY_BYTES = 64
_SQLITE_BUSY_TIMEOUT_SAFETY_MILLISECONDS = 5
_T = TypeVar("_T")


def _platform_name() -> str:
    """Resolve platform through a narrow contract-test seam."""
    return os.name


class SqliteLifecycleAuthority:
    """One SQLite authority with method-scoped, process-owned connections."""

    capabilities = AuthorityCapabilities(durable=True, multiprocess=True)
    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "transaction_scope": "authority",
        "exact_cas": True,
        "portable_query": True,
        "canonical_scan": True,
        "index_acceleration": False,
    }

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
        self._bootstrap_lock = Lock()
        self._closed = False
        self.open_write_transactions = 0
        self._monotonic_clock: Callable[[], float] | None = None
        self._transaction_hook: Callable[[str], None] | None = None
        self._bootstrap_hook: Callable[[str], None] | None = None

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
        return self._now() + self.lifecycle_limits.authority_busy_timeout_seconds

    def _now(self) -> float:
        """Resolve the production clock lazily while retaining an explicit test seam."""
        clock = time.monotonic if self._monotonic_clock is None else self._monotonic_clock
        return clock()

    def _deadline_timeout(
        self,
        *,
        stage: str,
        started_at: float,
        deadline: float | None = None,
    ) -> CacheBlobLifecycleTimeoutError:
        now = self._now()
        elapsed = max(0.0, now - started_at)
        remaining = None if deadline is None else max(0.0, deadline - now)
        return CacheBlobLifecycleTimeoutError(
            "Lifecycle authority busy deadline expired",
            context={
                "operation": "lifecycle_authority",
                "stage": stage,
                "elapsed_seconds": elapsed,
                "remaining_seconds": remaining,
                "authority_busy_timeout_seconds": (
                    self.lifecycle_limits.authority_busy_timeout_seconds
                ),
                "authority_path": str(self.path),
                "retryable": True,
            },
        )

    def _remaining_for_stage(
        self,
        deadline: float,
        *,
        stage: str,
        started_at: float,
    ) -> float:
        remaining = deadline - self._now()
        if remaining <= 0:
            raise self._deadline_timeout(
                stage=stage,
                started_at=started_at,
                deadline=deadline,
            )
        return remaining

    @staticmethod
    def _busy_timeout_milliseconds(remaining_seconds: float) -> int:
        """Reserve scheduler overhead while never extending a caller deadline."""
        return max(
            0,
            int(remaining_seconds * 1000) - _SQLITE_BUSY_TIMEOUT_SAFETY_MILLISECONDS,
        )

    @staticmethod
    def _set_busy_timeout(
        connection: sqlite3.Connection,
        milliseconds: int,
    ) -> None:
        """Apply one generated SQLite busy budget without parameter interpolation."""
        if type(milliseconds) is not int or milliseconds < 0:
            raise ValueError("SQLite busy timeout must be a non-negative integer")
        connection.execute(f"PRAGMA busy_timeout = {milliseconds}")

    def _apply_stage_busy_timeout(
        self,
        connection: sqlite3.Connection,
        *,
        deadline: float,
        started_at: float,
        stage: str,
    ) -> None:
        """Refresh SQLite's local wait budget without extending one deadline."""
        remaining = self._remaining_for_stage(
            deadline,
            stage=stage,
            started_at=started_at,
        )
        try:
            self._set_busy_timeout(
                connection,
                self._busy_timeout_milliseconds(remaining),
            )
        except sqlite3.Error as error:
            self._translate_sqlite_error(
                error,
                operation="lifecycle_authority",
                stage=stage,
                deadline=deadline,
                started_at=started_at,
            )

    def _execute_for_stage(
        self,
        connection: sqlite3.Connection,
        statement: str,
        *,
        parameters: object = (),
        deadline: float,
        started_at: float,
        stage: str,
    ) -> sqlite3.Cursor:
        """Run one potentially contended SQLite statement under the sole budget."""
        self._apply_stage_busy_timeout(
            connection,
            deadline=deadline,
            started_at=started_at,
            stage=stage,
        )
        try:
            return connection.execute(statement, parameters)
        except sqlite3.Error as error:
            self._translate_sqlite_error(
                error,
                operation="lifecycle_authority",
                stage=stage,
                deadline=deadline,
                started_at=started_at,
            )
            raise AssertionError("SQLite error translation must raise")

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

    def _has_only_new_authority_bootstrap_artifacts(self) -> bool:
        """Recognize the bounded payload-stage artifacts preceding O_EXCL leaf creation."""
        try:
            root_entries = tuple(self.root.iterdir())
        except FileNotFoundError:
            return False
        return self.path not in root_entries and all(
            entry.name in _BOOTSTRAP_ROOT_NAMES for entry in root_entries
        )

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

    def preflight_mutation(self) -> None:
        """Reject an unusable mutation topology before materializing authority state.

        In particular, a Windows root is an offline-provisioned trust boundary.
        This method intentionally performs only ownership and DACL/topology
        inspection; it must run before any read can cause SQLite bootstrap or
        any payload helper can create the configured root.
        """
        self._require_owned_open()
        self._validate_mutation_topology()

    def initialize(self) -> None:
        """Create/validate this catalog before starting shared workers.

        Existing incomplete or obsolete catalogs are never adopted or upgraded.
        Concurrent first creation is not a supported availability guarantee.
        """
        started_at = self._now()
        with self._connection(
            mutation=True, deadline=self._deadline(None), started_at=started_at,
        ):
            pass

    def _materialize_database_file(self, deadline: float, *, started_at: float) -> bool:
        """Create only the contained database leaf and report whether this won."""
        self._validate_mutation_topology()
        with self._bootstrap_lock:
            state = self._classify_for_open()
            if state == "authority":
                # Validate an existing database through a read-only handle
                # before a mutation-capable connection can configure SQLite.
                # Unsupported, foreign, or incomplete files therefore stay
                # byte-for-byte untouched, including journal sidecars.
                self._validate_existing_authority_readonly(
                    deadline=deadline,
                    started_at=started_at,
                )
                return False
            if state in {
                "wrong_root",
                "invalid_reserved_directory",
                "invalid_authority",
            }:
                self._reject_non_authority_state(state)
            if state == "established":
                if not (
                    self._is_pristine_reserved_bootstrap()
                    or self._has_only_new_authority_bootstrap_artifacts()
                ):
                    self._reject_non_authority_state(state)
                state = "ready"
            if state == "missing":
                self._reach_bootstrap_boundary("authority.bootstrap.before_root")
                try:
                    self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
                except FileExistsError:
                    state = self._classify_for_open()
                    if state == "authority":
                        return False
                    if state == "established" and (
                        self._is_pristine_reserved_bootstrap()
                        or self._has_only_new_authority_bootstrap_artifacts()
                    ):
                        state = "ready"
                    if state not in {"ready", "missing"}:
                        self._reject_non_authority_state(state)
            reserved = self.root / AUTHORITY_RELATIVE_PATH.parent
            if not reserved.exists():
                self._reach_bootstrap_boundary("authority.bootstrap.before_reserved")
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

            self._reach_bootstrap_boundary("authority.bootstrap.before_leaf")
            self._remaining_for_stage(
                deadline,
                stage="connection_open",
                started_at=started_at,
            )
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
        """Classify only SQLite primary BUSY/LOCKED codes, never error text."""
        error_code = getattr(error, "sqlite_errorcode", None)
        if type(error_code) is not int:
            return False
        return (error_code & 0xFF) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}

    def _translate_sqlite_error(
        self,
        error: sqlite3.Error,
        *,
        operation: str,
        stage: str,
        deadline: float,
        started_at: float,
    ) -> None:
        if operation != "lifecycle_authority":
            raise AssertionError("Lifecycle SQLite errors require canonical operation")
        if self._is_busy_error(error):
            raise self._deadline_timeout(
                stage=stage,
                started_at=started_at,
                deadline=deadline,
            ) from error
        primary = getattr(error, "sqlite_errorcode", None)
        primary = primary & 0xFF if type(primary) is int else None
        if primary in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}:
            # A regular leaf can still be hostile or corrupt rather than a
            # lifecycle authority. Treat it as migration-required evidence;
            # do not let connection configuration rewrite or adopt it.
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority database is incompatible",
                context={"authority_path": str(self.path), "operation": operation},
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

    def set_bootstrap_hook_for_test(self, hook: Callable[[str], None] | None) -> None:
        """Install a test-only observer for fresh-root bootstrap boundaries."""
        self._bootstrap_hook = hook

    def _reach_bootstrap_boundary(self, boundary: str) -> None:
        if self._bootstrap_hook is not None:
            self._bootstrap_hook(boundary)

    def _configure_connection(
        self,
        connection: sqlite3.Connection,
        *,
        initialize: bool,
        deadline: float,
        started_at: float,
    ) -> None:
        """Configure and read back every connection-level authority pragma."""
        if initialize:
            mode = self._execute_for_stage(
                connection,
                "PRAGMA journal_mode = DELETE",
                deadline=deadline,
                started_at=started_at,
                stage="connection_configure",
            ).fetchone()[0]
        else:
            mode = self._execute_for_stage(
                connection,
                "PRAGMA journal_mode",
                deadline=deadline,
                started_at=started_at,
                stage="connection_configure",
            ).fetchone()[0]
        self._execute_for_stage(
            connection,
            "PRAGMA synchronous = EXTRA",
            deadline=deadline,
            started_at=started_at,
            stage="connection_configure",
        )
        self._execute_for_stage(
            connection,
            "PRAGMA foreign_keys = ON",
            deadline=deadline,
            started_at=started_at,
            stage="connection_configure",
        )
        self._execute_for_stage(
            connection,
            "PRAGMA trusted_schema = OFF",
            deadline=deadline,
            started_at=started_at,
            stage="connection_configure",
        )

        synchronous = self._execute_for_stage(
            connection,
            "PRAGMA synchronous",
            deadline=deadline,
            started_at=started_at,
            stage="connection_configure",
        ).fetchone()[0]
        foreign_keys = self._execute_for_stage(
            connection,
            "PRAGMA foreign_keys",
            deadline=deadline,
            started_at=started_at,
            stage="connection_configure",
        ).fetchone()[0]
        trusted_schema = self._execute_for_stage(
            connection,
            "PRAGMA trusted_schema",
            deadline=deadline,
            started_at=started_at,
            stage="connection_configure",
        ).fetchone()[0]
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

    def _initialize_schema(
        self,
        connection: sqlite3.Connection,
        *,
        deadline: float,
        started_at: float,
    ) -> None:
        """Create the current format-2 authority schema exactly once."""
        self._apply_stage_busy_timeout(
            connection,
            deadline=deadline,
            started_at=started_at,
            stage="schema_initialize",
        )
        try:
            connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.Error as error:
            self._translate_sqlite_error(
                error,
                operation="lifecycle_authority",
                stage="schema_initialize",
                deadline=deadline,
                started_at=started_at,
            )
        self._reach_bootstrap_boundary("authority.schema_initialize.exclusive_acquired")
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS store_identity (identity TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entry_lineage (key TEXT PRIMARY KEY, lineage INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                "key TEXT PRIMARY KEY, generation TEXT NOT NULL, locator TEXT NOT NULL, "
                "manifest BLOB NOT NULL, manifest_digest TEXT NOT NULL, lineage INTEGER NOT NULL, "
                "revision INTEGER NOT NULL, transport_evidence BLOB)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS mutations ("
                "operation_id TEXT PRIMARY KEY, key TEXT NOT NULL, generation TEXT NOT NULL, "
                "locator TEXT NOT NULL, expected_lineage INTEGER, expected_revision INTEGER, "
                "expected_generation TEXT, expected_manifest_digest TEXT, manifest BLOB NOT NULL, "
                "verified_digest TEXT, verified_size INTEGER, transport_evidence BLOB, "
                "state TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS authority_state ("
                "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), revision INTEGER NOT NULL, "
                "projection_dirty INTEGER NOT NULL CHECK (projection_dirty IN (0, 1)), "
                "migration_run_id TEXT, migration_plan_digest TEXT, "
                "migration_candidate_digest TEXT, migration_source_revision INTEGER, "
                "migration_activated_revision INTEGER, "
                "migration_state TEXT NOT NULL DEFAULT 'idle' "
                "CHECK (migration_state IN ('idle', 'candidate', 'activated_offline', 'active', 'rolled_back')), "
                "migration_active_selection TEXT NOT NULL DEFAULT 'source' "
                "CHECK (migration_active_selection IN ('source', 'candidate', 'prior')), "
                "migration_rollback_eligible INTEGER NOT NULL DEFAULT 0 "
                "CHECK (migration_rollback_eligible IN (0, 1)))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS migration_store_entries ("
                "run_id TEXT NOT NULL, selection TEXT NOT NULL "
                "CHECK (selection IN ('candidate', 'prior')), key TEXT NOT NULL, "
                "generation TEXT NOT NULL, locator TEXT NOT NULL, manifest BLOB NOT NULL, "
                "manifest_digest TEXT NOT NULL, lineage INTEGER NOT NULL, "
                "entry_revision INTEGER NOT NULL, transport_evidence BLOB, "
                "PRIMARY KEY (run_id, selection, key))"
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
                "manifest BLOB NOT NULL, manifest_digest TEXT NOT NULL, "
                "transport_evidence BLOB, state TEXT NOT NULL, "
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
            connection.execute(f"PRAGMA application_id = {SQLITE_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {SQLITE_USER_VERSION}")
            connection.execute("COMMIT")
            self._reach_bootstrap_boundary("authority.schema_initialize.committed")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise


    def _validate_schema(
        self,
        connection: sqlite3.Connection,
        *,
        deadline: float,
        started_at: float,
    ) -> None:
        """Reject an unknown database before it participates in a transition."""
        application_id = self._execute_for_stage(
            connection,
            "PRAGMA application_id",
            deadline=deadline,
            started_at=started_at,
            stage="schema_validate",
        ).fetchone()[0]
        version = self._execute_for_stage(
            connection,
            "PRAGMA user_version",
            deadline=deadline,
            started_at=started_at,
            stage="schema_validate",
        ).fetchone()[0]
        if application_id != SQLITE_APPLICATION_ID:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority application ID is incompatible"
            )
        if version > SQLITE_USER_VERSION:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority schema version is unsupported"
            )
        if version != SQLITE_USER_VERSION:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority schema version is incompatible"
            )

        # Known development layouts shared version 1. Validate their required
        # columns without running DDL, even on the first mutation after reopen.
        try:
            connection.execute(
                "SELECT e.manifest_digest, e.lineage, e.revision, "
                "m.expected_generation, m.expected_manifest_digest, "
                "d.debt_id, d.role, c.last_key, t.manifest_digest, "
                "r.mutation_cursor, r.debt_cursor, a.action_id "
                "FROM entries e, mutations m, cleanup_debt d, clear_runs c, "
                "clear_targets t, reconciliation_runs r, reconciliation_actions a "
                "WHERE 0"
            )
        except sqlite3.OperationalError as error:
            if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_ERROR:
                raise CacheBlobMigrationRequiredError(
                    "Lifecycle authority layout requires explicit offline migration or rebuild"
                ) from error
            raise

        rows = self._execute_for_stage(
            connection,
            "SELECT identity FROM store_identity",
            deadline=deadline,
            started_at=started_at,
            stage="schema_validate",
        ).fetchall()
        if len(rows) != 1 or not isinstance(rows[0][0], str):
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority store identity is incompatible"
            )
        identity = rows[0][0]
        if not identity or len(identity.encode("utf-8")) > _MAX_STORE_IDENTITY_BYTES:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority store identity is incompatible"
            )

    def _validate_existing_authority_readonly(
        self,
        *,
        deadline: float,
        started_at: float,
    ) -> None:
        """Reject an established non-current layout without writable SQLite I/O."""
        try:
            connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro",
                uri=True,
                isolation_level=None,
                timeout=self._remaining_for_stage(
                    deadline,
                    stage="schema_validate",
                    started_at=started_at,
                ),
                check_same_thread=True,
            )
        except sqlite3.Error as error:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority layout requires explicit offline migration or rebuild"
            ) from error
        try:
            self._validate_schema(
                connection,
                deadline=deadline,
                started_at=started_at,
            )
        except CacheBlobMigrationRequiredError:
            raise
        except sqlite3.Error as error:
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority layout requires explicit offline migration or rebuild"
            ) from error
        finally:
            connection.close()

    @contextmanager
    def _connection(
        self,
        *,
        mutation: bool,
        deadline: float,
        started_at: float | None = None,
    ) -> Iterator[sqlite3.Connection | None]:
        """Open, harden, validate, and close one process/thread-owned connection."""
        self._require_owned_open()
        if started_at is None:
            started_at = self._now()
        created_new = False
        if mutation:
            created_new = self._materialize_database_file(
                deadline,
                started_at=started_at,
            )
        else:
            state = self._classify_for_open()
            if state == "established" and (
                self._is_pristine_reserved_bootstrap()
                or self._has_only_new_authority_bootstrap_artifacts()
            ):
                # A concurrent first writer may have made the reserved
                # namespace (and other bounded bootstrap artifacts) visible
                # before it wins the SQLite O_EXCL leaf. Treat that exact
                # state as absent for a read rather than rejecting it as
                # legacy evidence; no payload becomes readable until a later
                # authority-backed observation succeeds.
                state = "ready"
            if state in {"missing", "ready"}:
                yield None
                return
            if state != "authority":
                self._reject_non_authority_state(state)

        connection: sqlite3.Connection | None = None
        try:
            remaining = self._remaining_for_stage(
                deadline,
                stage="connection_open",
                started_at=started_at,
            )
            if mutation:
                try:
                    connection = sqlite3.connect(
                        self.path,
                        isolation_level=None,
                        timeout=remaining,
                        check_same_thread=True,
                    )
                except sqlite3.Error as error:
                    self._translate_sqlite_error(
                        error,
                        operation="lifecycle_authority",
                        stage="connection_open",
                        deadline=deadline,
                        started_at=started_at,
                    )
            else:
                try:
                    connection = sqlite3.connect(
                        f"{self.path.as_uri()}?mode=ro",
                        uri=True,
                        isolation_level=None,
                        timeout=remaining,
                        check_same_thread=True,
                    )
                except sqlite3.Error as error:
                    self._translate_sqlite_error(
                        error,
                        operation="lifecycle_authority",
                        stage="connection_open",
                        deadline=deadline,
                        started_at=started_at,
                    )
            assert connection is not None
            self._configure_connection(
                connection,
                initialize=created_new,
                deadline=deadline,
                started_at=started_at,
            )
            if created_new:
                self._initialize_schema(
                    connection,
                    deadline=deadline,
                    started_at=started_at,
                )
            self._validate_schema(
                connection, deadline=deadline, started_at=started_at,
            )
            if created_new:
                self._reach_bootstrap_boundary("authority.schema_ready.published")
            yield connection
        except (
            CacheBlobBackendError,
            CacheBlobLifecycleTimeoutError,
            CacheBlobMigrationRequiredError,
        ):
            raise
        except sqlite3.Error as error:
            self._translate_sqlite_error(
                error,
                operation="lifecycle_authority",
                stage="schema_validate",
                deadline=deadline,
                started_at=started_at,
            )
        finally:
            if connection is not None:
                connection.close()

    @contextmanager
    def _read_connection(
        self,
        *,
        deadline: float | None = None,
        started_at: float | None = None,
    ) -> Iterator[sqlite3.Connection | None]:
        """Yield a read-only connection under one caller-owned operation budget."""
        if started_at is None:
            started_at = self._now()
        absolute_deadline = self._deadline(deadline)
        with self._connection(
            mutation=False,
            deadline=absolute_deadline,
            started_at=started_at,
        ) as connection:
            if connection is None:
                yield None
                return
            yield connection

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
        uncertain_classifier: Callable[[float, float], _T] | None = None,
    ) -> _T:
        """Run one bounded state transition with rollback on every failure."""
        self._require_owned_open()
        started_at = self._now()
        absolute_deadline = self._deadline(deadline)
        with self._connection(
            mutation=True,
            deadline=absolute_deadline,
            started_at=started_at,
        ) as connection:
            assert connection is not None
            self._apply_stage_busy_timeout(
                connection,
                deadline=absolute_deadline,
                started_at=started_at,
                stage="sqlite_busy",
            )
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as error:
                self._translate_sqlite_error(
                    error,
                    operation="lifecycle_authority",
                    stage="sqlite_busy",
                    deadline=absolute_deadline,
                    started_at=started_at,
                )
                raise AssertionError("SQLite error translation must raise")
            with self._state_lock:
                self.open_write_transactions += 1
            try:
                result = callback(connection)
                self._reach_transaction_boundary("authority.transaction.before_commit")
                self._execute_for_stage(
                    connection,
                    "COMMIT",
                    deadline=absolute_deadline,
                    started_at=started_at,
                    stage="sqlite_busy",
                )
                self._reach_transaction_boundary("authority.transaction.committed")
                return result
            except BaseException as error:
                committed = not connection.in_transaction
                if connection.in_transaction:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.Error as rollback_error:
                        error.add_note(
                            "Lifecycle authority rollback failed after primary error: "
                            f"{rollback_error}"
                        )
                sqlite_error = (
                    error
                    if isinstance(error, sqlite3.Error)
                    else error.__cause__
                    if isinstance(error.__cause__, sqlite3.Error)
                    else None
                )
                if (
                    committed
                    and uncertain_classifier is not None
                    and isinstance(sqlite_error, sqlite3.Error)
                ):
                    return uncertain_classifier(absolute_deadline, started_at)
                raise
            finally:
                with self._state_lock:
                    self.open_write_transactions -= 1

    def read_entry(self, key: str) -> EntrySnapshot | None:
        with self._read_connection() as connection:
            if connection is None:
                return None
            row = connection.execute(
                "SELECT generation, locator, manifest, manifest_digest, lineage, revision, "
                "transport_evidence "
                "FROM entries WHERE key = ?",
                (key,),
            ).fetchone()
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
                    None if row[6] is None else bytes(row[6]),
                )
            except (TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "Lifecycle authority entry row is malformed",
                    context={"operation": "lifecycle_authority"},
                ) from error

    def read_expectation(self, key: str) -> EntryExpectation:
        """Read exact present or absent lineage without decoding a manifest."""
        with self._read_connection() as connection:
            if connection is None:
                return EntryExpectation.absent()
            return self._expectation(connection, key)

    def replace_committed_metadata(
        self,
        entry: EntrySnapshot,
        *,
        expected: EntryExpectation,
        manifest: bytes,
    ) -> EntrySnapshot:
        """Atomically replace signed mutable descriptor fields for one entry."""

        def replace_metadata(connection: sqlite3.Connection) -> EntrySnapshot:
            observed = self._expectation(connection, entry.key)
            if not self._matches(expected, observed):
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement expectation no longer matches authority"
                )
            row = connection.execute(
                "SELECT generation, locator, manifest, manifest_digest, lineage, revision, "
                "transport_evidence "
                "FROM entries WHERE key = ?",
                (entry.key,),
            ).fetchone()
            if row is None:
                raise CacheBlobLifecycleConflictError("Metadata replacement entry is absent")
            stored = EntrySnapshot(
                entry.key,
                row[0],
                row[1],
                bytes(row[2]),
                EntryExpectation(row[4], row[5], row[0], row[3]),
                None if row[6] is None else bytes(row[6]),
            )
            if stored != entry:
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement entry changed before comparison"
                )
            try:
                current_manifest = BlobManifest.from_canonical_bytes(stored.manifest)
                replacement_manifest = BlobManifest.from_canonical_bytes(manifest)
            except (CacheManifestIntegrityError, TypeError, ValueError) as error:
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement descriptor is malformed"
                ) from error
            if (
                replacement_manifest.key != current_manifest.key
                or replacement_manifest.generation != current_manifest.generation
                or replacement_manifest.locator != current_manifest.locator
                or replacement_manifest.digest != current_manifest.digest
                or replacement_manifest.byte_size != current_manifest.byte_size
                or replacement_manifest.state != current_manifest.state
            ):
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement changed immutable payload identity"
                )
            current_record = current_manifest.to_mapping()
            replacement_record = replacement_manifest.to_mapping()
            for field_name in (
                "catalog_values",
                "catalog_presence",
                "user_metadata",
                "signature",
            ):
                current_record.pop(field_name, None)
                replacement_record.pop(field_name, None)
            if current_record != replacement_record:
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement changed immutable descriptor fields"
                )
            revision = connection.execute(
                "SELECT revision FROM authority_state WHERE singleton = 1"
            ).fetchone()[0] + 1
            manifest_digest = hashlib.sha256(manifest).hexdigest()
            cursor = connection.execute(
                "UPDATE entries SET manifest = ?, manifest_digest = ?, revision = ? "
                "WHERE key = ? AND generation = ? AND locator = ? AND lineage = ? "
                "AND revision = ? AND manifest_digest = ?",
                (
                    manifest,
                    manifest_digest,
                    revision,
                    entry.key,
                    stored.generation,
                    stored.locator,
                    stored.expectation.lineage,
                    stored.expectation.revision,
                    stored.expectation.manifest_digest,
                ),
            )
            if cursor.rowcount != 1:
                raise CacheBlobLifecycleConflictError(
                    "Metadata replacement compare-and-swap did not apply"
                )
            connection.execute(
                "UPDATE authority_state SET revision = ?, projection_dirty = 1 "
                "WHERE singleton = 1",
                (revision,),
            )
            return EntrySnapshot(
                key=stored.key,
                generation=stored.generation,
                locator=stored.locator,
                manifest=bytes(manifest),
                expectation=EntryExpectation(
                    stored.expectation.lineage,
                    revision,
                    stored.generation,
                    manifest_digest,
                ),
                transport_evidence=stored.transport_evidence,
            )

        return self._transaction(replace_metadata)

    def prepare_mutation(self, spec: MutationSpec) -> PreparedMutation:
        # This observer is deliberately outside the mutation transaction: it
        # is a test-only schedule seam, and production authorities never
        # install it. Keeping it here lets independent fresh authorities
        # rendezvous before FIFO admission without allowing preflight work to
        # escape the admitted writer's deadline.
        if self._bootstrap_hook is not None and self._classify_for_open() == "missing":
            self._reach_bootstrap_boundary("authority.bootstrap.classified")

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
        if (
            prepared.spec.manifest
            and proof.manifest
            and prepared.spec.manifest != proof.manifest
        ):
            raise CacheBlobLifecycleConflictError(
                "Verification descriptor differs from prepared descriptor"
            )
        descriptor = proof.manifest or prepared.spec.manifest

        def record(connection: sqlite3.Connection) -> None:
            cursor = connection.execute(
                "UPDATE mutations SET verified_digest = ?, verified_size = ?, manifest = ?, "
                "transport_evidence = ? "
                "WHERE operation_id = ? AND state = 'prepared' AND manifest = ?",
                (
                    proof.digest,
                    proof.byte_size,
                    descriptor,
                    proof.transport_evidence,
                    prepared.operation_id,
                    prepared.spec.manifest,
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
                "expected_generation, expected_manifest_digest, manifest, verified_digest, "
                "transport_evidence, state "
                "FROM mutations WHERE operation_id = ?",
                (prepared.operation_id,),
            ).fetchone()
            if row is None:
                raise CacheBlobLifecycleConflictError("Mutation does not exist")
            if row[10] == "promoted":
                return self._promoted_result(connection, prepared.operation_id)
            if row[10] != "prepared" or row[8] is None:
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
                "INSERT INTO entries(key, generation, locator, manifest, manifest_digest, lineage, revision, "
                "transport_evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET "
                "generation=excluded.generation, locator=excluded.locator, "
                "manifest=excluded.manifest, manifest_digest=excluded.manifest_digest, "
                "lineage=excluded.lineage, "
                "revision=excluded.revision, transport_evidence=excluded.transport_evidence",
                (
                    row[0],
                    row[1],
                    row[2],
                    row[7],
                    manifest_digest,
                    next_lineage,
                    revision,
                    row[9],
                ),
            )
            self._reach_transaction_boundary("promote.after_entry")
            self._reach_transaction_boundary("promote.after_descriptor")
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
            self._reach_transaction_boundary("promote.after_revision")
            return self._promoted_result(connection, prepared.operation_id)

        return self._transaction(
            promote,
            uncertain_classifier=lambda deadline, started_at: self._classify_promoted_mutation(
                prepared,
                deadline=deadline,
                started_at=started_at,
            ),
        )

    def _promoted_result(
        self, connection: sqlite3.Connection, operation_id: str
    ) -> PromotionResult:
        """Read one already-committed promotion without reapplying it."""
        row = connection.execute(
            "SELECT m.key, e.generation, e.locator, e.manifest, e.manifest_digest, "
            "e.lineage, e.revision, e.transport_evidence FROM mutations AS m "
            "JOIN entries AS e ON e.key = m.key "
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
                None if row[7] is None else bytes(row[7]),
            ),
            tuple(CleanupDebt(*debt_row) for debt_row in debt_rows),
        )

    def read_mutation(self, operation_id: str) -> MutationReplay | None:
        """Read one exact replay record without changing SQLite authority state."""
        MutationSpec.validate_operation_id(operation_id)
        with self._read_connection() as connection:
            if connection is None:
                return None
            row = connection.execute(
                "SELECT key, generation, locator, expected_lineage, expected_revision, "
                "expected_generation, expected_manifest_digest, manifest, verified_digest, "
                "verified_size, transport_evidence, state FROM mutations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                return None
            try:
                spec = MutationSpec.create(
                    operation_id=operation_id,
                    key=row[0],
                    generation=row[1],
                    candidate_locator=row[2],
                    expected=EntryExpectation(row[3], row[4], row[5], row[6]),
                    manifest=bytes(row[7]),
                )
                prepared = PreparedMutation(operation_id, spec)
                if (row[8] is None) != (row[9] is None):
                    raise ValueError("verification proof fields are incomplete")
                proof = (
                    None
                    if row[8] is None
                    else VerificationProof(
                        row[8],
                        row[9],
                        spec.manifest,
                        None if row[10] is None else bytes(row[10]),
                    )
                )
                if row[11] == "prepared":
                    return MutationReplay(prepared, row[11], proof)
                if row[11] == "promoted":
                    return MutationReplay(
                        prepared,
                        row[11],
                        proof,
                        self._promoted_result(connection, operation_id),
                    )
            except (TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "SQLite lifecycle authority mutation row is malformed",
                    context={"operation": "lifecycle_authority_read_mutation"},
                ) from error
            raise CacheBlobLifecycleConflictError(
                "SQLite lifecycle operation is not replayable",
                context={"operation_id": operation_id, "state": row[11]},
            )

    def _classify_promoted_mutation(
        self,
        prepared: PreparedMutation,
        *,
        deadline: float,
        started_at: float,
    ) -> PromotionResult:
        """Resolve an ambiguous commit by reading exact durable operation state."""
        with self._read_connection(
            deadline=deadline,
            started_at=started_at,
        ) as connection:
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
        with self._read_connection() as connection:
            if connection is None:
                return ()
            rows = connection.execute(
                "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision, "
                "transport_evidence "
                "FROM entries ORDER BY key"
            ).fetchall()
            return tuple(
                EntrySnapshot(
                    row[0],
                    row[1],
                    row[2],
                    bytes(row[3]),
                    EntryExpectation(row[5], row[6], row[1], row[4]),
                    None if row[7] is None else bytes(row[7]),
                )
                for row in rows
            )

    @staticmethod
    def _inventory_identity(connection: sqlite3.Connection) -> AuthorityIdentitySnapshot:
        """Read one validated SQLite authority identity inside the caller's snapshot."""
        identity_row = connection.execute("SELECT identity FROM store_identity").fetchone()
        revision_row = connection.execute(
            "SELECT revision FROM authority_state WHERE singleton = 1"
        ).fetchone()
        if (
            identity_row is None
            or revision_row is None
            or not isinstance(identity_row[0], str)
            or type(revision_row[0]) is not int
        ):
            raise CacheBlobBackendError(
                "SQLite migration inventory identity is malformed",
                context={"operation": "migration_inventory"},
            )
        return AuthorityIdentitySnapshot(
            store_id=identity_row[0],
            revision=revision_row[0],
            authority_kind="sqlite",
            capability=f"sqlite-lifecycle-authority-v{SQLITE_USER_VERSION}",
            schema_version=SQLITE_USER_VERSION,
        )

    def identity_snapshot(self) -> AuthorityIdentitySnapshot:
        """Return one read-only SQLite identity without creating or upgrading a store."""
        with self._read_connection() as connection:
            if connection is None:
                raise CacheBlobMigrationRequiredError(
                    "SQLite migration inventory requires an initialized authority"
                )
            connection.execute("BEGIN")
            try:
                identity = self._inventory_identity(connection)
                connection.execute("COMMIT")
                return identity
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def inventory_page(
        self,
        cursor: AuthorityInventoryCursor | None = None,
        *,
        limit: int | None = None,
        work_cap: int | None = None,
    ) -> AuthorityInventoryPage:
        """Read one raw, revision-bound keyset page in one SQLite read transaction."""
        effective_limit, effective_work_cap = validate_inventory_page_request(
            limit=limit,
            work_cap=work_cap,
            default_limit=self.lifecycle_limits.manifest_page_size,
            default_work_cap=self.lifecycle_limits.max_operation_record_bytes,
        )
        with self._read_connection() as connection:
            if connection is None:
                raise CacheBlobMigrationRequiredError(
                    "SQLite migration inventory requires an initialized authority"
                )
            connection.execute("BEGIN")
            try:
                identity = self._inventory_identity(connection)
                if cursor is not None and (
                    cursor.store_id != identity.store_id
                    or cursor.revision != identity.revision
                ):
                    raise CacheBlobLifecycleConflictError(
                        "Migration inventory changed; reinspection is required"
                    )
                if cursor is None or cursor.last_key is None:
                    rows = connection.execute(
                        "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision, "
                        "transport_evidence "
                        "FROM entries ORDER BY key, generation LIMIT ?",
                        (effective_limit + 1,),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision, "
                        "transport_evidence "
                        "FROM entries WHERE key > ? OR (key = ? AND generation > ?) "
                        "ORDER BY key, generation LIMIT ?",
                        (
                            cursor.last_key,
                            cursor.last_key,
                            cursor.last_generation,
                            effective_limit + 1,
                        ),
                    ).fetchall()
                entries: list[EntrySnapshot] = []
                work_seen = 0
                for row in rows:
                    if len(entries) == effective_limit:
                        break
                    try:
                        snapshot = EntrySnapshot(
                            key=row[0],
                            generation=row[1],
                            locator=row[2],
                            manifest=bytes(row[3]),
                            expectation=EntryExpectation(row[5], row[6], row[1], row[4]),
                            transport_evidence=(
                                None if row[7] is None else bytes(row[7])
                            ),
                        )
                    except (TypeError, ValueError) as error:
                        raise CacheBlobBackendError(
                            "SQLite migration inventory row is malformed",
                            context={"operation": "migration_inventory"},
                        ) from error
                    entry_bytes = len(snapshot.manifest)
                    if work_seen + entry_bytes > effective_work_cap:
                        if not entries:
                            raise CacheBlobBackendError(
                                "Migration inventory entry exceeds the configured work bound",
                                context={"operation": "migration_inventory"},
                            )
                        break
                    entries.append(snapshot)
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
                page = AuthorityInventoryPage(
                    identity=identity,
                    entries=tuple(entries),
                    next_cursor=next_cursor,
                    exhausted=exhausted,
                )
                connection.execute("COMMIT")
                return page
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _candidate_entries_from_rows(
        rows: list[tuple[object, ...]],
    ) -> tuple[AuthorityInventoryEntry, ...]:
        """Reconstruct exact candidate descriptors from authority-owned rows."""
        entries: list[AuthorityInventoryEntry] = []
        for row in rows:
            try:
                manifest_bytes = bytes(row[3])
                manifest = BlobManifest.from_canonical_bytes(manifest_bytes)
                entry = AuthorityInventoryEntry(
                    key=str(row[0]),
                    generation=str(row[1]),
                    locator=str(row[2]),
                    manifest=manifest_bytes,
                    payload_digest=manifest.digest,
                    byte_size=manifest.byte_size,
                )
            except (TypeError, ValueError) as error:
                raise CacheBlobBackendError(
                    "SQLite migration candidate row is malformed",
                    context={"operation": "migration_candidate"},
                ) from error
            if (
                manifest.key != entry.key
                or manifest.generation != entry.generation
                or manifest.locator != entry.locator
                or manifest.canonical_bytes() != manifest_bytes
                or hashlib.sha256(manifest_bytes).hexdigest() != row[4]
            ):
                raise CacheBlobBackendError(
                    "SQLite migration candidate row does not corroborate its manifest",
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
        """Reject incomplete or stale external verification before a transaction writes state."""
        if receipt.destination_identity != identity or receipt.destination_revision != identity.revision:
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

    @staticmethod
    def _publication_state_row(connection: sqlite3.Connection) -> tuple[object, ...]:
        """Load all whole-store publication facts from the singleton authority row."""
        row = connection.execute(
            "SELECT revision, migration_run_id, migration_plan_digest, "
            "migration_candidate_digest, migration_source_revision, "
            "migration_activated_revision, migration_state, migration_active_selection, "
            "migration_rollback_eligible FROM authority_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise CacheBlobBackendError(
                "SQLite migration publication state is missing",
                context={"operation": "migration_publication"},
            )
        return tuple(row)

    def record_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> VerifiedCandidateReceipt:
        """Record only a complete, externally verified candidate in one authority transaction."""
        if not isinstance(receipt, VerifiedCandidateReceipt) or not isinstance(entries, tuple):
            raise TypeError("migration candidate receipt and entries must be immutable values")

        def record(connection: sqlite3.Connection) -> VerifiedCandidateReceipt:
            identity = self._inventory_identity(connection)
            self._validate_verified_candidate(identity, receipt, entries)
            state_row = self._publication_state_row(connection)
            state = AuthorityPublicationState(state_row[6])
            if state is AuthorityPublicationState.CANDIDATE:
                if state_row[1] == receipt.run_id and state_row[2] == receipt.plan_digest and state_row[4] == receipt.source_revision:
                    rows = connection.execute(
                        "SELECT key, generation, locator, manifest, manifest_digest "
                        "FROM migration_store_entries WHERE run_id = ? AND selection = 'candidate' "
                        "ORDER BY key",
                        (receipt.run_id,),
                    ).fetchall()
                    stored = self._candidate_entries_from_rows(rows)
                    if stored == entries:
                        return receipt
                    if stored == entries[: len(stored)]:
                        for entry in entries[len(stored) :]:
                            evidence_row = connection.execute(
                                "SELECT transport_evidence FROM entries "
                                "WHERE key = ? AND generation = ? AND locator = ? "
                                "AND manifest_digest = ?",
                                (
                                    entry.key,
                                    entry.generation,
                                    entry.locator,
                                    entry.manifest_digest,
                                ),
                            ).fetchone()
                            connection.execute(
                                "INSERT INTO migration_store_entries("
                                "run_id, selection, key, generation, locator, manifest, manifest_digest, "
                                "lineage, entry_revision, transport_evidence) "
                                "VALUES (?, 'candidate', ?, ?, ?, ?, ?, 0, ?, ?)",
                                (
                                    receipt.run_id,
                                    entry.key,
                                    entry.generation,
                                    entry.locator,
                                    entry.manifest,
                                    entry.manifest_digest,
                                    receipt.source_revision,
                                    None if evidence_row is None else evidence_row[0],
                                ),
                            )
                        connection.execute(
                            "UPDATE authority_state SET migration_candidate_digest = ? "
                            "WHERE singleton = 1 AND migration_run_id = ? AND migration_plan_digest = ? "
                            "AND migration_state = 'candidate'",
                            (receipt.candidate_digest, receipt.run_id, receipt.plan_digest),
                        )
                        return receipt
                raise CacheBlobLifecycleConflictError(
                    "A different verified migration candidate is already recorded"
                )
            if state is not AuthorityPublicationState.IDLE or state_row[1] is not None:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate recording requires an unselected active authority"
                )
            for entry in entries:
                evidence_row = connection.execute(
                    "SELECT transport_evidence FROM entries "
                    "WHERE key = ? AND generation = ? AND locator = ? "
                    "AND manifest_digest = ?",
                    (
                        entry.key,
                        entry.generation,
                        entry.locator,
                        entry.manifest_digest,
                    ),
                ).fetchone()
                connection.execute(
                    "INSERT INTO migration_store_entries("
                    "run_id, selection, key, generation, locator, manifest, manifest_digest, "
                    "lineage, entry_revision, transport_evidence) "
                    "VALUES (?, 'candidate', ?, ?, ?, ?, ?, 0, ?, ?)",
                    (
                        receipt.run_id,
                        entry.key,
                        entry.generation,
                        entry.locator,
                        entry.manifest,
                        entry.manifest_digest,
                        receipt.source_revision,
                        None if evidence_row is None else evidence_row[0],
                    ),
                )
            connection.execute(
                "UPDATE authority_state SET migration_run_id = ?, migration_plan_digest = ?, "
                "migration_candidate_digest = ?, migration_source_revision = ?, "
                "migration_activated_revision = NULL, migration_state = 'candidate', "
                    "migration_active_selection = 'source', migration_rollback_eligible = 0 "
                "WHERE singleton = 1",
                (
                    receipt.run_id,
                    receipt.plan_digest,
                    receipt.candidate_digest,
                    receipt.source_revision,
                ),
            )
            return receipt

        return self._transaction(record)

    def candidate_entries_for_run(self, *, run_id: str) -> tuple[AuthorityInventoryEntry, ...]:
        """Return only this authority's durably attributed candidate descriptors."""
        with self._read_connection() as connection:
            if connection is None:
                raise CacheBlobMigrationRequiredError(
                    "SQLite migration candidate evidence requires an initialized authority"
                )
            connection.execute("BEGIN")
            try:
                state_row = self._publication_state_row(connection)
                if state_row[1] != run_id or AuthorityPublicationState(state_row[6]) not in {
                    AuthorityPublicationState.CANDIDATE,
                    AuthorityPublicationState.ACTIVATED_OFFLINE,
                    AuthorityPublicationState.ACTIVE,
                    AuthorityPublicationState.ROLLED_BACK,
                }:
                    connection.execute("COMMIT")
                    return ()
                rows = connection.execute(
                    "SELECT key, generation, locator, manifest, manifest_digest "
                    "FROM migration_store_entries WHERE run_id = ? AND selection = 'candidate' "
                    "ORDER BY key",
                    (run_id,),
                ).fetchall()
                entries = self._candidate_entries_from_rows(rows)
                connection.execute("COMMIT")
                return entries
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    def discard_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> None:
        """Clear one exact unactivated candidate after external retirement succeeds."""
        if not isinstance(receipt, VerifiedCandidateReceipt) or not isinstance(entries, tuple):
            raise TypeError("migration candidate receipt and entries must be immutable values")

        def discard(connection: sqlite3.Connection) -> None:
            identity = self._inventory_identity(connection)
            self._validate_verified_candidate(identity, receipt, entries)
            state_row = self._publication_state_row(connection)
            if (
                AuthorityPublicationState(state_row[6]) is not AuthorityPublicationState.CANDIDATE
                or state_row[1] != receipt.run_id
                or state_row[2] != receipt.plan_digest
                or state_row[3] != receipt.candidate_digest
            ):
                raise CacheBlobLifecycleConflictError(
                    "Only the exact unactivated migration candidate may be discarded"
                )
            rows = connection.execute(
                "SELECT key, generation, locator, manifest, manifest_digest, transport_evidence "
                "FROM migration_store_entries WHERE run_id = ? AND selection = 'candidate' "
                "ORDER BY key",
                (receipt.run_id,),
            ).fetchall()
            if self._candidate_entries_from_rows(rows) != entries:
                raise CacheBlobLifecycleConflictError(
                    "Migration candidate entries changed before discard"
                )
            connection.execute(
                "DELETE FROM migration_store_entries WHERE run_id = ? AND selection = 'candidate'",
                (receipt.run_id,),
            )
            connection.execute(
                "UPDATE authority_state SET migration_run_id = NULL, migration_plan_digest = NULL, "
                "migration_candidate_digest = NULL, migration_source_revision = NULL, "
                "migration_activated_revision = NULL, migration_state = 'idle', "
                "migration_active_selection = 'source', migration_rollback_eligible = 0 "
                "WHERE singleton = 1",
            )

        self._transaction(discard)

    def activate_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> ActivationReceipt:
        """Atomically select the recorded whole candidate and retain exact prior rows."""
        if not isinstance(receipt, VerifiedCandidateReceipt) or not isinstance(entries, tuple):
            raise TypeError("migration candidate receipt and entries must be immutable values")

        def activate(connection: sqlite3.Connection) -> ActivationReceipt:
            state_row = self._publication_state_row(connection)
            state = AuthorityPublicationState(state_row[6])
            if state is AuthorityPublicationState.ACTIVATED_OFFLINE:
                if (
                    state_row[1] == receipt.run_id
                    and state_row[2] == receipt.plan_digest
                    and state_row[3] == receipt.candidate_digest
                    and type(state_row[5]) is int
                ):
                    return ActivationReceipt(
                        candidate_receipt=receipt,
                        activation_revision=state_row[5],
                        prior_store=PriorStoreReceipt(
                            run_id=receipt.run_id,
                            revision=state_row[0] - 1,
                            entry_count=connection.execute(
                                "SELECT count(*) FROM migration_store_entries "
                                "WHERE run_id = ? AND selection = 'prior'",
                                (receipt.run_id,),
                            ).fetchone()[0],
                        ),
                    )
                raise CacheBlobLifecycleConflictError(
                    "A different migration activation is already selected"
                )
            if state is not AuthorityPublicationState.CANDIDATE or (
                state_row[1] != receipt.run_id
                or state_row[2] != receipt.plan_digest
                or state_row[3] != receipt.candidate_digest
                or state_row[4] != receipt.source_revision
            ):
                raise CacheBlobLifecycleConflictError(
                    "Verified migration candidate is not recorded by this authority"
                )
            candidate_rows = connection.execute(
                "SELECT key, generation, locator, manifest, manifest_digest, transport_evidence "
                "FROM migration_store_entries WHERE run_id = ? AND selection = 'candidate' "
                "ORDER BY key",
                (receipt.run_id,),
            ).fetchall()
            stored = self._candidate_entries_from_rows(
                [row[:5] for row in candidate_rows]
            )
            if stored != entries:
                raise CacheBlobLifecycleConflictError(
                    "Recorded migration candidate differs from verified external receipt"
                )
            current_entries = connection.execute(
                "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision, "
                "transport_evidence "
                "FROM entries ORDER BY key"
            ).fetchall()
            for prior in current_entries:
                connection.execute(
                    "INSERT INTO migration_store_entries("
                    "run_id, selection, key, generation, locator, manifest, manifest_digest, "
                    "lineage, entry_revision, transport_evidence) "
                    "VALUES (?, 'prior', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (receipt.run_id, *prior),
                )
            next_revision = state_row[0] + 1
            connection.execute("DELETE FROM entries")
            candidate_evidence = {row[0]: row[5] for row in candidate_rows}
            for candidate in entries:
                previous = connection.execute(
                    "SELECT lineage FROM entry_lineage WHERE key = ?", (candidate.key,)
                ).fetchone()
                lineage = (previous[0] if previous is not None else 0) + 1
                connection.execute(
                    "INSERT INTO entry_lineage(key, lineage) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET lineage = excluded.lineage",
                    (candidate.key, lineage),
                )
                connection.execute(
                    "INSERT INTO entries(key, generation, locator, manifest, manifest_digest, "
                    "lineage, revision, transport_evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        candidate.key,
                        candidate.generation,
                        candidate.locator,
                        candidate.manifest,
                        candidate.manifest_digest,
                        lineage,
                        next_revision,
                        candidate_evidence[candidate.key],
                    ),
                )
            connection.execute(
                "UPDATE authority_state SET revision = ?, projection_dirty = 1, "
                "migration_activated_revision = ?, migration_state = 'activated_offline', "
                "migration_active_selection = 'candidate', migration_rollback_eligible = 1 "
                "WHERE singleton = 1",
                (next_revision, next_revision),
            )
            return ActivationReceipt(
                candidate_receipt=receipt,
                activation_revision=next_revision,
                prior_store=PriorStoreReceipt(
                    run_id=receipt.run_id,
                    revision=state_row[0],
                    entry_count=len(current_entries),
                ),
            )

        return self._transaction(activate)

    def publication_state(self) -> AuthorityPublicationState:
        """Return narrow maintenance status without admitting ordinary workers."""
        with self._read_connection() as connection:
            if connection is None:
                return AuthorityPublicationState.IDLE
            return AuthorityPublicationState(self._publication_state_row(connection)[6])

    def activation_receipt_for_candidate(
        self, receipt: VerifiedCandidateReceipt
    ) -> ActivationReceipt | None:
        """Classify a lost activation response without inferring state from candidate paths."""
        if not isinstance(receipt, VerifiedCandidateReceipt):
            raise TypeError("receipt must be a VerifiedCandidateReceipt")
        with self._read_connection() as connection:
            if connection is None:
                return None
            state_row = self._publication_state_row(connection)
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
            return ActivationReceipt(
                candidate_receipt=receipt,
                activation_revision=state_row[5],
                prior_store=PriorStoreReceipt(
                    run_id=receipt.run_id,
                    revision=state_row[0] - 1,
                    entry_count=connection.execute(
                        "SELECT count(*) FROM migration_store_entries "
                        "WHERE run_id = ? AND selection = 'prior'",
                        (receipt.run_id,),
                    ).fetchone()[0],
                ),
            )

    def require_ordinary_worker_access(self) -> None:
        """Seal all normal BlobStore work until offline rollback or finalize."""
        if self.publication_state() is AuthorityPublicationState.ACTIVATED_OFFLINE:
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "Offline migration activation requires explicit rollback or finalize before workers restart",
                context={"operation": "migration.worker_access"},
            )

    def rollback_verified_candidate(self, *, run_id: str) -> RollbackReceipt:
        """Restore retained prior rows in one authority transaction while workers remain stopped."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")

        def rollback(connection: sqlite3.Connection) -> RollbackReceipt:
            state_row = self._publication_state_row(connection)
            if (
                AuthorityPublicationState(state_row[6])
                is not AuthorityPublicationState.ACTIVATED_OFFLINE
                or state_row[1] != run_id
                or state_row[8] != 1
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration rollback requires the selected offline activation"
                )
            prior_rows = connection.execute(
                "SELECT key, generation, locator, manifest, manifest_digest, lineage, "
                "transport_evidence "
                "FROM migration_store_entries WHERE run_id = ? AND selection = 'prior' ORDER BY key",
                (run_id,),
            ).fetchall()
            next_revision = state_row[0] + 1
            connection.execute("DELETE FROM entries")
            for prior in prior_rows:
                connection.execute(
                    "INSERT INTO entry_lineage(key, lineage) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET lineage = excluded.lineage",
                    (prior[0], prior[5]),
                )
                connection.execute(
                    "INSERT INTO entries(key, generation, locator, manifest, manifest_digest, "
                    "lineage, revision, transport_evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (*prior[:6], next_revision, prior[6]),
                )
            connection.execute(
                "UPDATE authority_state SET revision = ?, projection_dirty = 1, "
                "migration_state = 'rolled_back', migration_active_selection = 'prior', "
                "migration_rollback_eligible = 0 WHERE singleton = 1",
                (next_revision,),
            )
            return RollbackReceipt(run_id=run_id, rollback_revision=next_revision)

        return self._transaction(rollback)

    def finalize_verified_candidate(self, *, run_id: str) -> FinalizeReceipt:
        """End rollback eligibility without deleting the separately retained prior copy."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")

        def finalize(connection: sqlite3.Connection) -> FinalizeReceipt:
            state_row = self._publication_state_row(connection)
            if (
                AuthorityPublicationState(state_row[6])
                is not AuthorityPublicationState.ACTIVATED_OFFLINE
                or state_row[1] != run_id
                or state_row[8] != 1
            ):
                raise CacheBlobLifecycleConflictError(
                    "Migration finalize requires the selected offline activation"
                )
            connection.execute(
                "UPDATE authority_state SET migration_state = 'active', "
                "migration_active_selection = 'candidate', migration_rollback_eligible = 0 "
                "WHERE singleton = 1"
            )
            return FinalizeReceipt(run_id=run_id, finalized_revision=state_row[0])

        return self._transaction(finalize)

    def retained_prior_entries(self, *, run_id: str) -> tuple[EntrySnapshot, ...]:
        """Return exact retained rows only for this finalized migration run."""
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")

        def retained(connection: sqlite3.Connection) -> tuple[EntrySnapshot, ...]:
            state_row = self._publication_state_row(connection)
            if (
                AuthorityPublicationState(state_row[6]) is not AuthorityPublicationState.ACTIVE
                or state_row[1] != run_id
                or state_row[8] != 0
            ):
                raise CacheBlobLifecycleConflictError(
                    "Retained prior entries require the finalized selected migration"
                )
            rows = connection.execute(
                "SELECT key, generation, locator, manifest, manifest_digest, lineage, entry_revision, "
                "transport_evidence "
                "FROM migration_store_entries WHERE run_id = ? AND selection = 'prior' "
                "ORDER BY key, generation",
                (run_id,),
            ).fetchall()
            return tuple(
                EntrySnapshot(
                    row[0],
                    row[1],
                    row[2],
                    bytes(row[3]),
                    EntryExpectation(row[5], row[6], row[1], row[4]),
                    None if row[7] is None else bytes(row[7]),
                )
                for row in rows
            )

        return self._transaction(retained)

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
        """Read a bounded canonical keyset page without a catalog mirror.

        SQLite only enumerates current authority identities.  Predicate
        evaluation happens after the BlobStore authenticates each signed
        descriptor, so this method never interpolates caller supplied fields
        or operators into SQL.
        """
        validate_catalog_page_request(
            query,
            schema=schema,
            cursor=cursor,
            limit=limit,
            work_cap=work_cap,
        )
        if cursor is not None:
            CatalogCursor.inspect(cursor, signing_key=signing_key)
        with self._read_connection() as connection:
            if connection is None:
                if cursor is not None:
                    raise CatalogCursorError("Catalog cursor does not match an absent authority")
                return CatalogPage((), 0, None, True)
            connection.execute("BEGIN")
            try:
                store_id = connection.execute(
                    "SELECT identity FROM store_identity"
                ).fetchone()[0]
                revision = connection.execute(
                    "SELECT revision FROM authority_state WHERE singleton = 1"
                ).fetchone()[0]
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
                    rows = connection.execute(
                        "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision, "
                        "transport_evidence "
                        "FROM entries ORDER BY key, generation LIMIT ?",
                        (work_cap + 1,),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT key, generation, locator, manifest, manifest_digest, lineage, revision, "
                        "transport_evidence "
                        "FROM entries WHERE key > ? OR (key = ? AND generation > ?) "
                        "ORDER BY key, generation LIMIT ?",
                        (
                            cursor_identity[0],
                            cursor_identity[0],
                            cursor_identity[1],
                            work_cap + 1,
                        ),
                    ).fetchall()
                snapshots = tuple(
                    EntrySnapshot(
                        row[0],
                        row[1],
                        row[2],
                        bytes(row[3]),
                        EntryExpectation(row[5], row[6], row[1], row[4]),
                        None if row[7] is None else bytes(row[7]),
                    )
                    for row in rows
                )
                page = page_from_canonical_scan(
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
                connection.execute("COMMIT")
                return page
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

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
        with self._read_connection() as connection:
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
        with self._read_connection() as connection:
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
                "manifest_digest, transport_evidence, state) "
                "SELECT ?, key, lineage, revision, generation, locator, manifest, "
                "manifest_digest, transport_evidence, 'pending' FROM entries",
                (token.value,),
            )
            return token

        return self._transaction(begin)

    def page_clear(self, token: PageToken) -> tuple[EntrySnapshot, ...]:
        with self._read_connection() as connection:
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
                "SELECT key, generation, locator, manifest, manifest_digest, lineage, entry_revision, "
                "transport_evidence "
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
                    None if row[7] is None else bytes(row[7]),
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
        with self._read_connection() as connection:
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
        with self._read_connection() as connection:
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
        with self._read_connection() as connection:
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
        started_at = self._now()
        absolute_deadline = self._deadline(None)
        snapshot_path: Path | None = None
        primary_error: BaseException | None = None
        try:
            with self._connection(
                mutation=False,
                deadline=absolute_deadline,
                started_at=started_at,
            ) as source:
                if source is None:
                    raise CacheBlobMigrationRequiredError(
                        "Lifecycle authority is absent for projection export"
                    )
                try:
                    self._execute_for_stage(
                        source,
                        "BEGIN",
                        deadline=absolute_deadline,
                        started_at=started_at,
                        stage="projection_backup",
                    )
                    revision = self._execute_for_stage(
                        source,
                        "SELECT revision FROM authority_state WHERE singleton = 1",
                        deadline=absolute_deadline,
                        started_at=started_at,
                        stage="projection_backup",
                    ).fetchone()
                    if revision is None or type(revision[0]) is not int:
                        raise CacheBlobMigrationRequiredError(
                            "Lifecycle authority projection revision is incompatible"
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
                        def progress(status: int, _remaining: int, _total: int) -> None:
                            primary_code = status & 0xFF
                            if primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                                error = sqlite3.OperationalError(
                                    "Lifecycle authority projection backup is busy"
                                )
                                error.sqlite_errorcode = primary_code
                                raise error
                            self._remaining_for_stage(
                                absolute_deadline,
                                stage="projection_backup",
                                started_at=started_at,
                            )

                        source.backup(
                            destination,
                            pages=16,
                            progress=progress,
                            sleep=0,
                        )
                    finally:
                        destination.close()
                except BaseException as error:
                    primary_error = error
                    raise
                finally:
                    if source.in_transaction:
                        try:
                            source.execute("ROLLBACK")
                        except sqlite3.Error as rollback_error:
                            if primary_error is not None:
                                primary_error.add_note(
                                    "Lifecycle authority projection rollback failed after "
                                    f"primary error: {rollback_error}"
                                )
                            else:
                                raise
            self._reach_transaction_boundary("projection.backup.closed_source")
            yield ProjectionBackup(snapshot_path, ProjectionRevision(revision[0]))
        except sqlite3.Error as error:
            self._translate_sqlite_error(
                error,
                operation="lifecycle_authority",
                stage="projection_backup",
                deadline=absolute_deadline,
                started_at=started_at,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if snapshot_path is not None:
                try:
                    snapshot_path.unlink()
                except FileNotFoundError:
                    pass

    def snapshot_state(self) -> AuthorityStateSnapshot:
        """Return one bounded authority-state diagnostic without exposing tables."""
        with self._read_connection() as connection:
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
        with self._read_connection() as connection:
            if connection is None:
                raise CacheBlobMigrationRequiredError("Lifecycle authority is absent")
            identity = connection.execute("SELECT identity FROM store_identity").fetchone()[0]
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

    def close(self) -> None:
        """Close this instance boundary without changing durable authority state."""
        with self._state_lock:
            self._closed = True


__all__ = [
    "AUTHORITY_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "SQLITE_APPLICATION_ID",
    "SQLITE_USER_VERSION",
    "SqliteLifecycleAuthority",
]
