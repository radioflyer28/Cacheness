"""Durable exact-byte persistence and bounded paging for lifecycle evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from heapq import nsmallest
from pathlib import Path
from threading import RLock, local
from typing import BinaryIO, Protocol

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheManifestIntegrityError,
    CacheReason,
    CacheUnsafePathError,
)

from .coordination import interprocess_open_file_lock, lock_stripe_index
from .manifest import MAX_MANIFEST_BYTES
from .operation_record import (
    MAX_CLEAR_TARGET_REFERENCE_CHUNKS,
    LifecycleOperationRecord,
)
from .path_security import ManagedFileOps, resolve_managed_locator, validate_blob_id


_CONDITIONAL_LOCK_STRIPES = tuple(RLock() for _ in range(64))
_OPERATION_LEASES = local()


@dataclass(frozen=True)
class OperationCursor:
    """Opaque stable position after one operation-record page."""

    operation_id: str

    def __post_init__(self) -> None:
        validate_blob_id(self.operation_id)


@dataclass(frozen=True)
class OperationPage:
    """One bounded, lexically ordered page of opaque evidence bytes."""

    entries: tuple[tuple[str, bytes], ...]
    next_cursor: OperationCursor | None


@dataclass(frozen=True)
class ReconciliationCheckpointCursor:
    """Opaque stable position after one reconciliation-sidecar page."""

    operation_id: str

    def __post_init__(self) -> None:
        validate_blob_id(self.operation_id)


@dataclass(frozen=True)
class ReconciliationCheckpointPage:
    """Bounded opaque inventory of private reconciliation sidecars."""

    entries: tuple[tuple[str, bytes | None], ...]
    next_cursor: ReconciliationCheckpointCursor | None


@dataclass(frozen=True)
class PendingControlCursor:
    """Opaque durable scheduling position for digest-bound pending controls."""

    name: str


@dataclass(frozen=True)
class PendingControlPage:
    """A bounded page of exact pending control candidates and their bytes."""

    entries: tuple[tuple[str, bytes | None], ...]
    next_cursor: PendingControlCursor | None


class OperationRecordRepository(Protocol):
    """Exact-byte evidence storage used by lifecycle and recovery code."""

    lifecycle_limits: LifecycleLimits

    def create_exclusive(
        self, record: LifecycleOperationRecord, raw_record: bytes
    ) -> Path:
        """Durably create evidence before a managed candidate side effect."""

    def get_raw(self, operation_id: str) -> bytes | None:
        """Return exact evidence bytes, or ``None`` only for an absent record."""

    def checkpoint_if_exact(
        self,
        record: LifecycleOperationRecord,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Checkpoint only if the exact previously observed bytes still exist."""

    def retire_if_exact(
        self, record: LifecycleOperationRecord, *, expected_raw: bytes
    ) -> None:
        """Retire evidence only after the exact terminal record remains current."""

    def list_page(
        self,
        cursor: OperationCursor | None = None,
        *,
        page_size: int | None = None,
    ) -> OperationPage:
        """Return a stable, bounded inventory page without interpreting evidence."""

    def list_reconciliation_checkpoint_page(
        self,
        cursor: ReconciliationCheckpointCursor | None = None,
        *,
        page_size: int | None = None,
    ) -> ReconciliationCheckpointPage:
        """Return bounded private-sidecar bytes without assigning authority."""

    def create_clear_target_page_exclusive(
        self, operation_id: str, page_id: str, raw_record: bytes
    ) -> Path:
        """Persist one exact clear target page before advancing its cursor."""

    def get_clear_target_page_raw(
        self, operation_id: str, page_id: str
    ) -> bytes | None:
        """Return one exact target page without assigning it authority."""

    def create_clear_target_checkpoint_exclusive(
        self, operation_id: str, page_id: str, raw_record: bytes
    ) -> Path:
        """Persist zero progress before destructive target work begins."""

    def get_clear_target_checkpoint_raw(
        self, operation_id: str, page_id: str
    ) -> bytes | None:
        """Return one exact clear checkpoint without interpreting progress."""

    def checkpoint_clear_target_if_exact(
        self,
        operation_id: str,
        page_id: str,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Advance target/page progress only from the exact observed bytes."""


class FileOperationRecordRepository:
    """Local operation evidence repository inside the managed store root."""

    def __init__(self, file_ops: ManagedFileOps, *, lifecycle_limits: LifecycleLimits):
        self.file_ops = file_ops
        # Retain the one caller-owned policy object; no field copies are used.
        self.lifecycle_limits = lifecycle_limits
        self._lock_handle_guard = RLock()
        self._lock_handles: dict[str, tuple[Path, BinaryIO, tuple[int, int]]] = {}

    def close(self) -> None:
        """Release retained bounded evidence-lock descriptors on store close."""
        first_error: Exception | None = None
        with self._lock_handle_guard:
            # Do not discard a descriptor from retry bookkeeping until its
            # close has a known successful outcome.  A failed close may leave
            # a process-scoped advisory lock live, so a later BlobStore.close
            # must retry rather than reporting a fictional terminal state.
            for lock_identity, (_locator, handle, _identity) in tuple(
                self._lock_handles.items()
            ):
                try:
                    handle.close()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                    continue
                self._lock_handles.pop(lock_identity, None)
        if first_error is not None:
            raise first_error

    def _conditional_lock_for(self, operation_id: str) -> RLock:
        """Return a shared bounded in-process stripe for one evidence ID.

        The stripe is deliberately module-wide so independently constructed
        repositories in the same process cannot bypass their common advisory
        file lock.  It only orders evidence transitions; it is never an
        authority substitute for the cross-process compare-and-mutate lock.
        """
        stripe = lock_stripe_index(self.file_ops.root, operation_id)
        return _CONDITIONAL_LOCK_STRIPES[stripe]

    def _operation_lease_key(self, operation_id: str) -> tuple[tuple[int, int], str]:
        """Identify one descriptor-root-scoped lease in this calling thread."""
        return self.file_ops.root_identity, operation_id

    @staticmethod
    def _held_operation_leases() -> dict[tuple[tuple[int, int], str], int]:
        """Return the current thread's reentrant operation-lease depths."""
        leases = getattr(_OPERATION_LEASES, "leases", None)
        if leases is None:
            leases = {}
            _OPERATION_LEASES.leases = leases
        return leases

    def _has_held_operation_lease(self, operation_id: str) -> bool:
        """Return whether this exact operation already owns its lease.

        Reentrancy is safe only for the same record and the same retained
        authority lock.  A parent clear record is not an authority lease for a
        child delete record: treating it as one loses the child's cross-process
        exact-CAS exclusion.
        """
        return self._held_operation_leases().get(self._operation_lease_key(operation_id), 0) > 0

    def _conditional_lock_locator(self, operation_id: str) -> Path:
        """Return one of a fixed number of durable evidence-CAS lock stripes."""
        stripe = lock_stripe_index(self.file_ops.root, operation_id)
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".conditional-locks" / f"{stripe:02x}.lock",
            operation="operation_record_conditional_lock",
            allow_missing_leaf=True,
        )

    def _retained_conditional_lock(
        self, lock_identity: str
    ) -> tuple[Path, BinaryIO, tuple[int, int]]:
        """Return one retained descriptor for a fixed bounded lock stripe."""
        return self._retained_lock(
            lock_identity, self._conditional_lock_locator(lock_identity)
        )

    def _retained_lock(
        self, lock_identity: str, locator: Path
    ) -> tuple[Path, BinaryIO, tuple[int, int]]:
        """Create and retain one single-linked managed control-lock descriptor."""
        # Operation IDs intentionally share a fixed number of lock stripes.
        # Cache by the contained stripe pathname, not the caller's identity, so
        # a long-lived store retains only the bounded stripe set.
        cache_identity = str(locator.relative_to(self.file_ops.root))
        with self._lock_handle_guard:
            cached = self._lock_handles.get(cache_identity)
            if cached is not None:
                return cached
            expected = self.file_ops.ensure_lifecycle_lock(locator)
            handle = self.file_ops.open_verified_regular_file(locator)
            try:
                observed = (os.fstat(handle.fileno()).st_dev, os.fstat(handle.fileno()).st_ino)
                if observed != expected:
                    raise CacheBlobLifecycleConflictError(
                        "Lifecycle evidence lock changed during descriptor retention",
                        context={"operation": "conditional_evidence"},
                    )
            except BaseException:
                handle.close()
                raise
            cached = (locator, handle, expected)
            self._lock_handles[cache_identity] = cached
            return cached

    def _clear_resume_lock_locator(self) -> Path:
        """Return the one store-wide clear-resume lease, distinct from CAS stripes."""
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".clear-resume.lock",
            operation="clear_resume_lock",
            allow_missing_leaf=True,
        )

    @contextmanager
    def clear_operation_transition(self, operation_id: str) -> Iterator[None]:
        """Serialize clear continuation without inheriting child record authority.

        Clear is the one aggregate operation allowed to use a store-wide lease.
        It guards resume/recovery of the clear control flow only; each target's
        delete/checkpoint still takes its own exact evidence lease.
        """
        del operation_id
        lock_identity = "clear-resume"
        with self._conditional_lock_for(lock_identity):
            locator, handle, expected_identity = self._retained_lock(
                lock_identity, self._clear_resume_lock_locator()
            )
            self.file_ops.assert_retained_lock_identity(locator, expected_identity)
            with interprocess_open_file_lock(
                handle, exclusive=True, operation="clear_resume"
            ):
                self.file_ops.assert_retained_lock_identity(locator, expected_identity)
                yield

    @contextmanager
    def _conditional_transition(
        self, transition_id: str, *, operation_id: str | None = None
    ) -> Iterator[None]:
        """Serialize one exact evidence CAS across repository objects/processes.

        ``flock`` is acquired only around read/compare/replace-or-delete.  It
        never spans handler serialization, manifest publication, or unrelated
        operation IDs, preserving the phase's no-global-normal-lock contract.
        """
        if operation_id is not None and self._has_held_operation_lease(operation_id):
            # This is true reentrancy for the exact record whose advisory lock
            # is already held.  No other operation ID can bypass the file lock.
            yield
            return

        lock_identity = (
            f"operation:{operation_id}" if operation_id is not None else transition_id
        )
        with self._conditional_lock_for(lock_identity):
            lock_locator, handle, expected_identity = self._retained_conditional_lock(
                lock_identity
            )
            self.file_ops.assert_retained_lock_identity(lock_locator, expected_identity)
            with interprocess_open_file_lock(
                handle,
                exclusive=True,
                operation="conditional_evidence",
            ):
                self.file_ops.assert_retained_lock_identity(
                    lock_locator, expected_identity
                )
                yield

    @contextmanager
    def operation_transition(self, operation_id: str) -> Iterator[None]:
        """Hold a narrow cross-process lease for one resumable operation."""
        key = self._operation_lease_key(operation_id)
        leases = self._held_operation_leases()
        if leases.get(key, 0):
            leases[key] += 1
            try:
                yield
            finally:
                leases[key] -= 1
            return

        if any(
            leased_root == self.file_ops.root_identity and depth > 0
            for (leased_root, _leased_operation_id), depth in leases.items()
        ):
            raise CacheBlobLifecycleConflictError(
                "Nested lifecycle operation leases require releasing the parent first",
                context={"operation_id": operation_id, "operation": "operation_transition"},
            )

        with self._conditional_transition(
            f"operation:{operation_id}", operation_id=operation_id
        ):
            leases[key] = 1
            try:
                yield
            finally:
                del leases[key]

    def _read_bounded(self, locator: Path, *, operation: str) -> bytes | None:
        """Read exact evidence only after enforced descriptor-bounded limits."""
        maximum = self.lifecycle_limits.max_operation_record_bytes
        try:
            size = self.file_ops.get_size(locator)
            if size < 0:
                return None
            if size > maximum:
                raise CacheManifestIntegrityError(
                    "Lifecycle control evidence exceeds the configured byte limit",
                    reason=CacheReason.MANIFEST_BOUNDS,
                )
            return self.file_ops.read_bytes_bounded(locator, max_bytes=maximum)
        except ValueError as exc:
            raise CacheManifestIntegrityError(
                "Lifecycle control evidence exceeds the configured byte limit",
                reason=CacheReason.MANIFEST_BOUNDS,
            ) from exc
        except FileNotFoundError:
            return None
        except CacheManifestIntegrityError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle control evidence could not be read",
                context={"operation": operation},
            ) from exc

    def locator_for(self, operation_id: str) -> Path:
        """Derive a contained locator from an opaque operation identifier."""
        safe_operation_id = validate_blob_id(operation_id)
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / f"{safe_operation_id}.json",
            operation="operation_record",
            allow_missing_leaf=True,
        )

    def _clear_target_locator(
        self, operation_id: str, page_id: str, *, checkpoint: bool
    ) -> Path:
        """Derive one contained deterministic clear page/checkpoint locator."""
        safe_operation_id = validate_blob_id(operation_id)
        safe_page_id = validate_blob_id(page_id)
        kind = "checkpoint" if checkpoint else "page"
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations")
            / f"clear-target-{kind}-{safe_operation_id}-{safe_page_id}.json",
            operation=f"clear_target_{kind}",
            allow_missing_leaf=True,
        )

    def _clear_target_reference_locator(
        self, operation_id: str, reference: str
    ) -> Path:
        """Derive a contained immutable exact-manifest sidecar locator."""
        safe_operation_id = validate_blob_id(operation_id)
        safe_reference = validate_blob_id(reference)
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations")
            / f"clear-target-reference-{safe_operation_id}-{safe_reference}.json",
            operation="clear_target_reference",
            allow_missing_leaf=True,
        )

    def _clear_target_reference_chunk_locator(
        self, operation_id: str, reference: str, chunk_index: int
    ) -> Path:
        """Derive one ordered bounded chunk of exact manifest evidence."""
        safe_operation_id = validate_blob_id(operation_id)
        safe_reference = validate_blob_id(reference)
        if type(chunk_index) is not int or not 0 <= chunk_index < MAX_CLEAR_TARGET_REFERENCE_CHUNKS:
            raise ValueError("clear target reference chunk index is invalid")
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations")
            / (
                "clear-target-reference-"
                f"{safe_operation_id}-{safe_reference}-part-{chunk_index:04x}.json"
            ),
            operation="clear_target_reference_chunk",
            allow_missing_leaf=True,
        )

    def clear_target_page_locator(self, operation_id: str, page_id: str) -> Path:
        """Return the exact contained locator for one target page."""
        return self._clear_target_locator(operation_id, page_id, checkpoint=False)

    def clear_target_checkpoint_locator(self, operation_id: str, page_id: str) -> Path:
        """Return the exact contained locator for one target checkpoint."""
        return self._clear_target_locator(operation_id, page_id, checkpoint=True)

    def clear_target_reference_locator(self, operation_id: str, reference: str) -> Path:
        """Return the exact contained locator for one referenced manifest copy."""
        return self._clear_target_reference_locator(operation_id, reference)

    def clear_target_reference_chunk_locator(
        self, operation_id: str, reference: str, chunk_index: int
    ) -> Path:
        """Return the deterministic contained locator for one reference chunk."""
        return self._clear_target_reference_chunk_locator(
            operation_id, reference, chunk_index
        )

    def reconciliation_checkpoint_locator(self, operation_id: str) -> Path:
        """Return a private sidecar used to resume one reconciliation action.

        The name deliberately cannot satisfy the 32-hex operation inventory
        grammar, so it never becomes lifecycle evidence or consumes a normal
        recovery page slot.
        """
        safe_operation_id = validate_blob_id(operation_id)
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / f"reconcile-action-{safe_operation_id}.json",
            operation="reconciliation_checkpoint",
            allow_missing_leaf=True,
        )

    def _pending_recovery_cursor_locator(self) -> Path:
        """Return private, non-authoritative progress for pending control scans."""
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".pending-recovery.cursor",
            operation="pending_recovery_cursor",
            allow_missing_leaf=True,
        )

    def _pending_recovery_cursor(self) -> PendingControlCursor | None:
        """Read bounded scheduler progress; malformed progress safely restarts."""
        try:
            raw = self.file_ops.read_bytes_bounded(
                self._pending_recovery_cursor_locator(),
                max_bytes=self.lifecycle_limits.max_operation_field_bytes,
            )
        except (FileNotFoundError, OSError, ValueError):
            return None
        try:
            name = raw.decode("ascii")
        except UnicodeDecodeError:
            return None
        return PendingControlCursor(name) if self._is_eligible_pending_name(name) else None

    def _checkpoint_pending_recovery_cursor(
        self, cursor: PendingControlCursor | None
    ) -> None:
        """Durably advance opaque scan scheduling without granting authority."""
        locator = self._pending_recovery_cursor_locator()
        if cursor is None:
            try:
                self.file_ops.delete_durable(locator)
            except FileNotFoundError:
                pass
            return
        self.file_ops.write_bytes_durable(locator, cursor.name.encode("ascii"))

    def get_reconciliation_checkpoint_raw(self, operation_id: str) -> bytes | None:
        """Read opaque reconciliation progress without granting it authority."""
        return self._read_bounded(
            self.reconciliation_checkpoint_locator(operation_id),
            operation="get_reconcile",
        )

    def list_reconciliation_checkpoint_page(
        self,
        cursor: ReconciliationCheckpointCursor | None = None,
        *,
        page_size: int | None = None,
    ) -> ReconciliationCheckpointPage:
        """Return one stable bounded page of reconciliation sidecar bytes.

        Sidecars are opaque scheduling evidence until the reconciler validates
        their exact canonical bytes.  The repository therefore advances past
        malformed candidates instead of repeatedly reading an invalid lexical
        prefix, and reports an unreadable/oversized candidate as ``None``.
        """
        if cursor is not None and not isinstance(cursor, ReconciliationCheckpointCursor):
            raise TypeError("reconciliation checkpoint cursor is invalid")
        limit = self._page_size(page_size)
        operations_directory = resolve_managed_locator(
            self.file_ops.root,
            "operations",
            operation="list_reconciliation_checkpoints",
            allow_missing_leaf=True,
        )
        prefix = "reconcile-action-"
        suffix = ".json"
        try:
            operation_ids = nsmallest(
                limit + 1,
                (
                    operation_id
                    for name in (path.name for path in operations_directory.iterdir())
                    if name.startswith(prefix)
                    and name.endswith(suffix)
                    for operation_id in (name[len(prefix) : -len(suffix)],)
                    if self._is_hex_identifier(operation_id)
                    and (cursor is None or operation_id > cursor.operation_id)
                ),
            )
        except FileNotFoundError:
            return ReconciliationCheckpointPage(entries=(), next_cursor=None)
        except OSError as exc:
            raise CacheBlobBackendError(
                "Reconciliation checkpoint directory could not be listed",
                context={"operation": "list_reconciliation_checkpoints"},
            ) from exc
        has_more = len(operation_ids) > limit
        page_ids = operation_ids[:limit]
        records: list[tuple[str, bytes | None]] = []
        for operation_id in page_ids:
            try:
                raw = self.get_reconciliation_checkpoint_raw(operation_id)
            except (CacheManifestIntegrityError, CacheBlobBackendError):
                raw = None
            records.append((operation_id, raw))
        return ReconciliationCheckpointPage(
            entries=tuple(records),
            next_cursor=(
                ReconciliationCheckpointCursor(page_ids[-1])
                if has_more and page_ids
                else None
            ),
        )

    def list_reconciliation_checkpoint_raws(self) -> tuple[tuple[str, bytes], ...]:
        """Compatibility iterator composed from bounded sidecar pages."""
        cursor: ReconciliationCheckpointCursor | None = None
        records: list[tuple[str, bytes]] = []
        while True:
            page = self.list_reconciliation_checkpoint_page(cursor)
            records.extend(
                (operation_id, raw)
                for operation_id, raw in page.entries
                if raw is not None
            )
            if page.next_cursor is None:
                return tuple(records)
            cursor = page.next_cursor

    def create_reconciliation_checkpoint_exclusive(
        self, operation_id: str, raw_record: bytes
    ) -> Path:
        """Durably persist action intent before a reconciler mutates storage."""
        if not isinstance(raw_record, bytes) or not raw_record:
            raise TypeError("Reconciliation checkpoints require non-empty bytes")
        try:
            return self.file_ops.create_bytes_durable_exclusive(
                self.reconciliation_checkpoint_locator(operation_id), raw_record
            )
        except FileExistsError as exc:
            raise CacheBlobLifecycleConflictError(
                "Reconciliation checkpoint already exists",
                context={"operation_id": operation_id, "operation": "create_reconcile"},
            ) from exc
        except OSError as exc:
            raise CacheBlobBackendError(
                "Reconciliation checkpoint could not be created",
                context={"operation_id": operation_id, "operation": "create_reconcile"},
            ) from exc

    def checkpoint_reconciliation_if_exact(
        self,
        operation_id: str,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Advance one reconciliation checkpoint only from exact bytes."""
        if not isinstance(expected_raw, bytes) or not isinstance(raw_record, bytes):
            raise TypeError("Reconciliation checkpoints require exact bytes")
        try:
            with self._conditional_transition(
                f"reconcile:{operation_id}", operation_id=operation_id
            ):
                current = self.get_reconciliation_checkpoint_raw(operation_id)
                if current != expected_raw:
                    raise CacheBlobLifecycleConflictError(
                        "Reconciliation checkpoint no longer matches",
                        context={
                            "operation_id": operation_id,
                            "operation": "checkpoint_reconcile",
                        },
                    )
                self.file_ops.write_bytes_durable(
                    self.reconciliation_checkpoint_locator(operation_id), raw_record
                )
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Reconciliation checkpoint could not be persisted",
                context={"operation_id": operation_id, "operation": "checkpoint_reconcile"},
            ) from exc

    def retire_reconciliation_checkpoint_if_exact(
        self, operation_id: str, *, expected_raw: bytes
    ) -> None:
        """Remove private progress only after the exact completed bytes remain."""
        try:
            with self._conditional_transition(
                f"reconcile:{operation_id}", operation_id=operation_id
            ):
                if self.get_reconciliation_checkpoint_raw(operation_id) != expected_raw:
                    raise CacheBlobLifecycleConflictError(
                        "Reconciliation checkpoint no longer matches",
                        context={
                            "operation_id": operation_id,
                            "operation": "retire_reconcile",
                        },
                    )
                self.file_ops.delete_durable(
                    self.reconciliation_checkpoint_locator(operation_id)
                )
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Reconciliation checkpoint could not be retired",
                context={"operation_id": operation_id, "operation": "retire_reconcile"},
            ) from exc

    def _create_clear_target_exclusive(
        self,
        operation_id: str,
        page_id: str,
        raw_record: bytes,
        *,
        checkpoint: bool,
    ) -> Path:
        """Create one durable clear control record without replacing evidence."""
        if not isinstance(raw_record, bytes) or not raw_record:
            raise TypeError("Clear target evidence must be non-empty bytes")
        locator = self._clear_target_locator(
            operation_id, page_id, checkpoint=checkpoint
        )
        try:
            return self.file_ops.create_bytes_durable_exclusive(locator, raw_record)
        except FileExistsError as exc:
            raise CacheBlobLifecycleConflictError(
                "Clear target evidence already exists",
                context={
                    "operation_id": operation_id,
                    "page_id": page_id,
                    "operation": "create_clear_target",
                },
            ) from exc
        except OSError as exc:
            raise CacheBlobBackendError(
                "Clear target evidence could not be created",
                context={
                    "operation_id": operation_id,
                    "page_id": page_id,
                    "operation": "create_clear_target",
                },
            ) from exc

    def create_clear_target_page_exclusive(
        self, operation_id: str, page_id: str, raw_record: bytes
    ) -> Path:
        """Durably save exact page targets before their source cursor advances."""
        return self._create_clear_target_exclusive(
            operation_id, page_id, raw_record, checkpoint=False
        )

    def create_clear_target_checkpoint_exclusive(
        self, operation_id: str, page_id: str, raw_record: bytes
    ) -> Path:
        """Durably save zero-progress state before a page can delete targets."""
        return self._create_clear_target_exclusive(
            operation_id, page_id, raw_record, checkpoint=True
        )

    def create_clear_target_reference_exclusive(
        self, operation_id: str, reference: str, raw_record: bytes
    ) -> Path:
        """Persist exact oversized target evidence without overwriting it."""
        if not isinstance(raw_record, bytes) or not raw_record:
            raise TypeError("Clear target reference requires non-empty bytes")
        if len(raw_record) > self.lifecycle_limits.max_operation_record_bytes:
            raise CacheManifestIntegrityError(
                "Clear target reference exceeds the configured byte limit",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        try:
            return self.file_ops.create_bytes_durable_exclusive(
                self._clear_target_reference_locator(operation_id, reference), raw_record
            )
        except FileExistsError as exc:
            raise CacheBlobLifecycleConflictError(
                "Clear target reference already exists",
                context={
                    "operation_id": operation_id,
                    "reference": reference,
                    "operation": "create_clear_target_reference",
                },
            ) from exc
        except OSError as exc:
            raise CacheBlobBackendError(
                "Clear target reference could not be created",
                context={
                    "operation_id": operation_id,
                    "reference": reference,
                    "operation": "create_clear_target_reference",
                },
            ) from exc

    def create_clear_target_reference_chunks_exclusive(
        self, operation_id: str, reference: str, raw_record: bytes
    ) -> int:
        """Persist bounded immutable chunks before a signed page resolves them.

        The caller must first prove the eventual page/control contract fits. A
        crash after a page is durable but before every chunk is written remains
        recoverable: pre-authority recovery authenticates and retires the page
        without using any target as deletion authority.
        """
        if not isinstance(raw_record, bytes) or not raw_record:
            raise TypeError("Clear target reference requires non-empty bytes")
        if len(raw_record) > MAX_MANIFEST_BYTES:
            raise CacheManifestIntegrityError(
                "Clear target reference exceeds the manifest byte limit",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        chunk_size = self.lifecycle_limits.max_operation_record_bytes
        chunk_count = (len(raw_record) + chunk_size - 1) // chunk_size
        if chunk_count > MAX_CLEAR_TARGET_REFERENCE_CHUNKS:
            raise CacheManifestIntegrityError(
                "Clear target reference needs too many bounded chunks",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        for chunk_index in range(chunk_count):
            chunk = raw_record[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
            locator = self._clear_target_reference_chunk_locator(
                operation_id, reference, chunk_index
            )
            try:
                self.file_ops.create_bytes_durable_exclusive(locator, chunk)
            except FileExistsError:
                existing = self._read_bounded(
                    locator, operation="create_clear_target_reference_chunk"
                )
                if existing != chunk:
                    raise CacheBlobLifecycleConflictError(
                        "Clear target reference chunk already exists",
                        context={
                            "operation_id": operation_id,
                            "reference": reference,
                            "chunk_index": str(chunk_index),
                            "operation": "create_clear_target_reference_chunk",
                        },
                    )
            except OSError as exc:
                raise CacheBlobBackendError(
                    "Clear target reference chunk could not be created",
                    context={
                        "operation_id": operation_id,
                        "reference": reference,
                        "chunk_index": str(chunk_index),
                        "operation": "create_clear_target_reference_chunk",
                    },
                ) from exc
        return chunk_count

    def _get_clear_target_raw(
        self, operation_id: str, page_id: str, *, checkpoint: bool
    ) -> bytes | None:
        """Load opaque exact clear evidence without declaring it authoritative."""
        return self._read_bounded(
            self._clear_target_locator(operation_id, page_id, checkpoint=checkpoint),
            operation="get_clear_target",
        )

    def get_clear_target_page_raw(
        self, operation_id: str, page_id: str
    ) -> bytes | None:
        """Read one exact target page without parsing or authenticating it."""
        return self._get_clear_target_raw(operation_id, page_id, checkpoint=False)

    def get_clear_target_checkpoint_raw(
        self, operation_id: str, page_id: str
    ) -> bytes | None:
        """Read one exact target checkpoint without parsing or authenticating it."""
        return self._get_clear_target_raw(operation_id, page_id, checkpoint=True)

    def get_clear_target_reference_raw(
        self,
        operation_id: str,
        reference: str,
        *,
        chunk_count: int | None = None,
        byte_length: int | None = None,
        chunk_digests: tuple[str, ...] | None = None,
    ) -> bytes | None:
        """Read exact target evidence under its signed bounded chunk contract.

        Legacy single-record references remain readable under the caller's
        limit. New references supply both count and length in the signed page;
        those values are validated before allocation or any chunk read.
        """
        if chunk_count is None and byte_length is None and chunk_digests is None:
            return self._read_bounded(
                self._clear_target_reference_locator(operation_id, reference),
                operation="get_clear_target_reference",
            )
        chunk_count, byte_length, chunk_digests = self._reference_chunk_contract(
            chunk_count, byte_length, chunk_digests
        )

        resolved = bytearray()
        for chunk_index in range(chunk_count):
            chunk = self._read_bounded(
                self._clear_target_reference_chunk_locator(
                    operation_id, reference, chunk_index
                ),
                operation="get_clear_target_reference_chunk",
            )
            if chunk is None:
                return None
            if (
                not chunk
                or hashlib.sha256(chunk).hexdigest() != chunk_digests[chunk_index]
                or len(resolved) + len(chunk) > byte_length
            ):
                raise CacheManifestIntegrityError(
                    "Clear target reference chunks do not match their signed length",
                    reason=CacheReason.MANIFEST_BOUNDS,
                )
            resolved.extend(chunk)
        if len(resolved) != byte_length:
            raise CacheManifestIntegrityError(
                "Clear target reference chunks do not match their signed length",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        return bytes(resolved)

    def _reference_chunk_contract(
        self,
        chunk_count: int | None,
        byte_length: int | None,
        chunk_digests: tuple[str, ...] | None,
    ) -> tuple[int, int, tuple[str, ...]]:
        """Validate signed reference bounds before reading or allocating chunks."""
        if (
            type(chunk_count) is not int
            or type(byte_length) is not int
            or not isinstance(chunk_digests, tuple)
            or not 0 < chunk_count <= MAX_CLEAR_TARGET_REFERENCE_CHUNKS
            or not 0 < byte_length <= MAX_MANIFEST_BYTES
            or len(chunk_digests) != chunk_count
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in chunk_digests
            )
        ):
            raise CacheManifestIntegrityError(
                "Clear target reference chunk contract is invalid",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        maximum = self.lifecycle_limits.max_operation_record_bytes
        if byte_length < chunk_count or byte_length > chunk_count * maximum:
            raise CacheManifestIntegrityError(
                "Clear target reference chunk contract exceeds configured bounds",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        return chunk_count, byte_length, chunk_digests

    def retire_clear_target_reference_chunks_if_bound(
        self,
        operation_id: str,
        reference: str,
        *,
        chunk_count: int | None,
        byte_length: int | None,
        chunk_digests: tuple[str, ...] | None,
    ) -> None:
        """Retire each page-bound chunk that still matches its signed digest.

        This is reserved for aborting an unauthoritative prepared inventory.
        A missing chunk is normal after an interrupted write; a mismatching
        chunk remains untouched rather than being inferred from its filename.
        """
        chunk_count, _byte_length, chunk_digests = self._reference_chunk_contract(
            chunk_count, byte_length, chunk_digests
        )
        with self._conditional_transition(
            f"clear-reference:{operation_id}:{reference}", operation_id=operation_id
        ):
            for chunk_index in range(chunk_count):
                locator = self._clear_target_reference_chunk_locator(
                    operation_id, reference, chunk_index
                )
                current = self._read_bounded(
                    locator, operation="abort_clear_target_reference_chunk"
                )
                if current is None:
                    continue
                if hashlib.sha256(current).hexdigest() != chunk_digests[chunk_index]:
                    raise CacheManifestIntegrityError(
                        "Clear target reference chunk no longer matches signed evidence"
                    )
                self.file_ops.delete_durable(locator)

    def _retire_locator_if_exact(
        self,
        locator: Path,
        *,
        transition_id: str,
        expected_raw: bytes,
        context: dict[str, str],
    ) -> bool:
        """Delete one control artifact only after exact-byte revalidation.

        A missing artifact is an idempotent completed retirement.  Any other
        bytes remain a typed conflict, never an invitation to delete by name.
        """
        if not isinstance(expected_raw, bytes):
            raise TypeError("Control artifact retirement requires exact bytes")
        with self._conditional_transition(
            transition_id, operation_id=context["operation_id"]
        ):
            current = self._read_bounded(locator, operation=context["operation"])
            if current is None:
                return False
            if current != expected_raw:
                raise CacheBlobLifecycleConflictError(
                    "Lifecycle control evidence no longer matches", context=context
                )
            self.file_ops.delete_durable(locator)
            return True

    def retire_clear_target_checkpoint_if_exact(
        self, operation_id: str, page_id: str, *, expected_raw: bytes
    ) -> bool:
        """Retire completed checkpoint evidence with exact CAS semantics."""
        return self._retire_locator_if_exact(
            self.clear_target_checkpoint_locator(operation_id, page_id),
            transition_id=f"clear:{operation_id}:{page_id}",
            expected_raw=expected_raw,
            context={
                "operation_id": operation_id,
                "page_id": page_id,
                "operation": "retire_clear_target_checkpoint",
            },
        )

    def retire_clear_target_page_if_exact(
        self, operation_id: str, page_id: str, *, expected_raw: bytes
    ) -> bool:
        """Retire completed target-page evidence with exact CAS semantics."""
        return self._retire_locator_if_exact(
            self.clear_target_page_locator(operation_id, page_id),
            transition_id=f"clear-page:{operation_id}:{page_id}",
            expected_raw=expected_raw,
            context={
                "operation_id": operation_id,
                "page_id": page_id,
                "operation": "retire_clear_target_page",
            },
        )

    def retire_clear_target_reference_if_exact(
        self,
        operation_id: str,
        reference: str,
        *,
        expected_raw: bytes,
        chunk_count: int | None = None,
        byte_length: int | None = None,
        chunk_digests: tuple[str, ...] | None = None,
    ) -> bool:
        """Retire authenticated reference bytes only after exact revalidation."""
        if chunk_count is None and byte_length is None and chunk_digests is None:
            return self._retire_locator_if_exact(
                self.clear_target_reference_locator(operation_id, reference),
                transition_id=f"clear-reference:{operation_id}:{reference}",
                expected_raw=expected_raw,
                context={
                    "operation_id": operation_id,
                    "reference": reference,
                    "operation": "retire_clear_target_reference",
                },
            )
        with self._conditional_transition(
            f"clear-reference:{operation_id}:{reference}", operation_id=operation_id
        ):
            current = self.get_clear_target_reference_raw(
                operation_id,
                reference,
                chunk_count=chunk_count,
                byte_length=byte_length,
                chunk_digests=chunk_digests,
            )
            if current is None:
                return False
            if current != expected_raw:
                raise CacheBlobLifecycleConflictError(
                    "Clear target reference no longer matches",
                    context={
                        "operation_id": operation_id,
                        "reference": reference,
                        "operation": "retire_clear_target_reference",
                    },
                )
            assert chunk_count is not None
            for chunk_index in range(chunk_count):
                self.file_ops.delete_durable(
                    self._clear_target_reference_chunk_locator(
                        operation_id, reference, chunk_index
                    )
                )
            return True

    def iter_clear_target_page_ids(self, operation_id: str) -> Iterator[str]:
        """Yield only this operation's syntactically exact page identifiers.

        Membership is never authority: callers must authenticate every bytes
        record before using the yielded ID to retire anything.  Streaming the
        directory avoids keeping a whole clear snapshot in memory during
        terminal cleanup and makes post-crash retirement resumable.
        """
        safe_operation_id = validate_blob_id(operation_id)
        prefix = f"clear-target-page-{safe_operation_id}-"
        operations_directory = resolve_managed_locator(
            self.file_ops.root,
            "operations",
            operation="list_clear_target_pages",
            allow_missing_leaf=True,
        )
        try:
            for path in operations_directory.iterdir():
                name = path.name
                if not name.startswith(prefix) or not name.endswith(".json"):
                    continue
                page_id = name[len(prefix) : -len(".json")]
                if len(page_id) != 32 or any(
                    character not in "0123456789abcdef" for character in page_id
                ):
                    continue
                yield page_id
        except FileNotFoundError:
            return
        except OSError as exc:
            raise CacheBlobBackendError(
                "Clear target evidence directory could not be listed",
                context={"operation_id": operation_id, "operation": "list_clear_pages"},
            ) from exc

    def checkpoint_clear_target_if_exact(
        self,
        operation_id: str,
        page_id: str,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Conditionally replace exact progress so stale writers cannot regress it."""
        if not isinstance(expected_raw, bytes) or not isinstance(raw_record, bytes):
            raise TypeError("Clear target checkpoints require exact bytes")
        locator = self.clear_target_checkpoint_locator(operation_id, page_id)
        try:
            with self._conditional_transition(
                f"clear:{operation_id}:{page_id}", operation_id=operation_id
            ):
                current = self.get_clear_target_checkpoint_raw(operation_id, page_id)
                if current != expected_raw:
                    raise CacheBlobLifecycleConflictError(
                        "Clear target checkpoint no longer matches",
                        context={
                            "operation_id": operation_id,
                            "page_id": page_id,
                            "operation": "checkpoint_clear_target_if_exact",
                        },
                    )
                self.file_ops.write_bytes_durable(locator, raw_record)
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Clear target checkpoint could not be persisted",
                context={
                    "operation_id": operation_id,
                    "page_id": page_id,
                    "operation": "checkpoint_clear_target_if_exact",
                },
            ) from exc

    def create_exclusive(
        self, record: LifecycleOperationRecord, raw_record: bytes
    ) -> Path:
        """Create evidence exclusively and durably before payload publication."""
        try:
            return self.file_ops.create_bytes_durable_exclusive(
                self.locator_for(record.operation_id), raw_record
            )
        except FileExistsError as exc:
            raise CacheBlobLifecycleConflictError(
                "Lifecycle operation evidence already exists",
                context={"operation_id": record.operation_id, "operation": "create"},
            ) from exc
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be created",
                context={"operation_id": record.operation_id, "operation": "create"},
            ) from exc

    def create(self, record: LifecycleOperationRecord, raw_record: bytes) -> Path:
        """Compatibility alias for explicit exclusive evidence creation."""
        return self.create_exclusive(record, raw_record)

    def get_raw(self, operation_id: str) -> bytes | None:
        """Read exact bytes without assigning evidence any read authority."""
        return self._read_bounded(
            self.locator_for(operation_id), operation="get_raw"
        )

    @staticmethod
    def _is_hex_identifier(value: str) -> bool:
        """Return whether ``value`` is the fixed opaque evidence identifier."""
        return len(value) == 32 and all(character in "0123456789abcdef" for character in value)

    def _recoverable_pending_final(self, name: str) -> tuple[Path, str | None] | None:
        """Resolve one strictly named lifecycle control final without guessing.

        Recovery only promotes candidates for the finite set of direct control
        records this repository creates.  Their payloads remain opaque here;
        digest-bound provenance and later authenticated lifecycle parsing supply
        the separate integrity boundaries.
        """
        if not name.endswith(".json"):
            return None
        base = name[:-5]
        if self._is_hex_identifier(base):
            return self.locator_for(base), base
        if base.startswith("reconcile-action-"):
            operation_id = base.removeprefix("reconcile-action-")
            if self._is_hex_identifier(operation_id):
                return self.reconciliation_checkpoint_locator(operation_id), None
            return None
        for kind in ("page", "checkpoint"):
            prefix = f"clear-target-{kind}-"
            if base.startswith(prefix):
                parts = base.removeprefix(prefix).split("-")
                if len(parts) == 2 and all(self._is_hex_identifier(part) for part in parts):
                    return (
                        self._clear_target_locator(
                            parts[0], parts[1], checkpoint=kind == "checkpoint"
                        ),
                        None,
                    )
                return None
        prefix = "clear-target-reference-"
        if not base.startswith(prefix):
            return None
        parts = base.removeprefix(prefix).split("-")
        if len(parts) == 2 and all(self._is_hex_identifier(part) for part in parts):
            return self._clear_target_reference_locator(parts[0], parts[1]), None
        if (
            len(parts) == 4
            and all(self._is_hex_identifier(part) for part in parts[:2])
            and parts[2] == "part"
            and len(parts[3]) == 4
            and all(character in "0123456789abcdef" for character in parts[3])
        ):
            chunk_index = int(parts[3], 16)
            if chunk_index < MAX_CLEAR_TARGET_REFERENCE_CHUNKS:
                return self._clear_target_reference_chunk_locator(
                    parts[0], parts[1], chunk_index
                ), None
        return None

    def list_pending_control_page(
        self, cursor: PendingControlCursor | None
    ) -> PendingControlPage:
        """Read at most one configured page of digest-bound pending controls."""
        operations_directory = resolve_managed_locator(
            self.file_ops.root,
            "operations",
            operation="list_pending_operation_records",
            allow_missing_leaf=True,
        )
        limit = self.lifecycle_limits.operation_page_size
        try:
            names = nsmallest(
                limit + 1,
                (
                    path.name
                    for path in operations_directory.iterdir()
                    if self._is_eligible_pending_name(path.name)
                    and (cursor is None or path.name > cursor.name)
                ),
            )
        except FileNotFoundError:
            return PendingControlPage(entries=(), next_cursor=None)
        except OSError as exc:
            raise CacheBlobBackendError(
                "Interrupted lifecycle control directory could not be listed",
                context={"operation": "recover_pending"},
            ) from exc
        has_more = len(names) > limit
        page_names = names[:limit]
        entries: list[tuple[str, bytes | None]] = []
        for name in page_names:
            pending = resolve_managed_locator(
                self.file_ops.root,
                Path("operations") / name,
                operation="recover_pending_operation_record",
            )
            try:
                raw = self.file_ops.read_bytes_bounded(
                    pending, max_bytes=self.lifecycle_limits.max_operation_record_bytes
                )
            except (FileNotFoundError, ValueError, OSError):
                raw = None
            entries.append((name, raw))
        return PendingControlPage(
            entries=tuple(entries),
            next_cursor=(
                PendingControlCursor(page_names[-1])
                if has_more and page_names
                else None
            ),
        )

    def recover_pending_operation_records(self) -> tuple[str, ...]:
        """Promote digest-bound interrupted lifecycle control candidates.

        A candidate name binds one exact final control name and SHA-256 of its
        contents. Malformed names, oversized bytes, and digest mismatches remain
        untouched for reconciliation reporting; no broad temporary-file sweep
        is ever used as ownership evidence.
        """
        cursor = self._pending_recovery_cursor()
        page = self.list_pending_control_page(cursor)
        recovered: list[str] = []
        eligible_actions = 0
        last_processed: str | None = None
        for name, raw in page.entries:
            if not (name.startswith(".") and name.endswith(".tmp")):
                last_processed = name
                continue
            pending_parts = name[1:-4].rsplit(".pending.", 1)
            if len(pending_parts) != 2:
                last_processed = name
                continue
            base, digest_and_token = pending_parts
            resolved_final = self._recoverable_pending_final(base)
            if resolved_final is None:
                last_processed = name
                continue
            final_locator, operation_id = resolved_final
            digest_parts = digest_and_token.rsplit(".", 1)
            if len(digest_parts) != 2:
                last_processed = name
                continue
            digest, token = digest_parts
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or len(token) != 32
                or any(character not in "0123456789abcdef" for character in token)
            ):
                last_processed = name
                continue
            if raw is None:
                last_processed = name
                continue
            if hashlib.sha256(raw).hexdigest() != digest:
                last_processed = name
                continue
            if eligible_actions >= self.lifecycle_limits.max_reconcile_actions:
                break
            try:
                promoted = self.file_ops.promote_durable_pending_control(
                    final_locator, raw, pending_name=name
                )
            except OSError as exc:
                raise CacheBlobBackendError(
                    "Interrupted lifecycle control evidence could not be recovered",
                    context={"operation_id": operation_id or "sidecar", "operation": "recover_pending"},
                ) from exc
            if promoted and operation_id is not None:
                recovered.append(operation_id)
            # A digest-valid candidate is real bounded recovery work even if
            # a concurrent winner already installed the same final record.
            # Syntax-valid bytes with the wrong digest deliberately do not
            # consume this budget and remain untouched/reportable.
            eligible_actions += 1
            last_processed = name
        # A pending candidate is scheduling evidence, never lifecycle
        # authority.  Persisting its opaque name lets later invocations move
        # past invalid or Windows-blocked prefixes without deleting them.
        next_cursor = (
            PendingControlCursor(last_processed)
            if last_processed is not None
            and (eligible_actions >= self.lifecycle_limits.max_reconcile_actions
                 or last_processed != page.entries[-1][0])
            else page.next_cursor
        )
        self._checkpoint_pending_recovery_cursor(next_cursor)
        return tuple(recovered)

    def _is_eligible_pending_name(self, name: str) -> bool:
        """Recognize only exact digest-bound pending evidence candidates.

        Inventory bounds apply after this grammar filter.  Ordinary files,
        malformed controls, and clear/reconciliation sidecars must never
        consume a lifecycle recovery action slot indefinitely.
        """
        if not (name.startswith(".") and name.endswith(".tmp")):
            return False
        pending_parts = name[1:-4].rsplit(".pending.", 1)
        if len(pending_parts) != 2:
            return False
        base, digest_and_token = pending_parts
        if self._recoverable_pending_final(base) is None:
            return False
        digest_parts = digest_and_token.rsplit(".", 1)
        if len(digest_parts) != 2:
            return False
        digest, token = digest_parts
        return (
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and len(token) == 32
            and all(character in "0123456789abcdef" for character in token)
        )

    def _require_exact_current(
        self, record: LifecycleOperationRecord, expected_raw: bytes, *, operation: str
    ) -> None:
        """Reject stale evidence without parsing or interpreting its bytes."""
        if self.get_raw(record.operation_id) != expected_raw:
            raise CacheBlobLifecycleConflictError(
                "Lifecycle operation evidence no longer matches",
                context={"operation_id": record.operation_id, "operation": operation},
            )

    def checkpoint_if_exact(
        self,
        record: LifecycleOperationRecord,
        *,
        expected_raw: bytes,
        raw_record: bytes,
    ) -> None:
        """Replace evidence only when the observed bytes remain current."""
        if not isinstance(expected_raw, bytes) or not isinstance(raw_record, bytes):
            raise TypeError("Operation evidence transitions require exact bytes")
        try:
            with self._conditional_transition(
                f"operation:{record.operation_id}", operation_id=record.operation_id
            ):
                self._require_exact_current(
                    record, expected_raw, operation="checkpoint_if_exact"
                )
                self.file_ops.write_bytes_durable(
                    self.locator_for(record.operation_id), raw_record
                )
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be checkpointed",
                context={
                    "operation_id": record.operation_id,
                    "operation": "checkpoint_if_exact",
                },
            ) from exc

    def checkpoint(self, record: LifecycleOperationRecord, raw_record: bytes) -> None:
        """Compatibility checkpoint that still verifies current evidence."""
        previous = self.get_raw(record.operation_id)
        if previous is None:
            raise CacheBlobLifecycleConflictError(
                "Lifecycle operation evidence is absent",
                context={"operation_id": record.operation_id, "operation": "checkpoint"},
            )
        self.checkpoint_if_exact(record, expected_raw=previous, raw_record=raw_record)

    def retire_if_exact(
        self, record: LifecycleOperationRecord, *, expected_raw: bytes
    ) -> None:
        """Retire evidence only when the exact terminal bytes remain current."""
        if not isinstance(expected_raw, bytes):
            raise TypeError("Operation evidence retirement requires exact bytes")
        try:
            with self._conditional_transition(
                f"operation:{record.operation_id}", operation_id=record.operation_id
            ):
                self._require_exact_current(
                    record, expected_raw, operation="retire_if_exact"
                )
                if not self.file_ops.delete_durable(self.locator_for(record.operation_id)):
                    raise CacheBlobLifecycleConflictError(
                        "Lifecycle operation evidence is absent",
                        context={
                            "operation_id": record.operation_id,
                            "operation": "retire_if_exact",
                        },
                    )
        except CacheBlobLifecycleConflictError:
            raise
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be retired",
                context={
                    "operation_id": record.operation_id,
                    "operation": "retire_if_exact",
                },
            ) from exc

    def retire(self, record: LifecycleOperationRecord) -> None:
        """Compatibility retirement that still compares current evidence."""
        previous = self.get_raw(record.operation_id)
        if previous is not None:
            self.retire_if_exact(record, expected_raw=previous)

    def _page_size(self, page_size: int | None) -> int:
        """Allow smaller caller pages but never bypass configured resource bounds."""
        resolved = (
            self.lifecycle_limits.operation_page_size
            if page_size is None
            else page_size
        )
        if type(resolved) is not int or resolved <= 0:
            raise ValueError("operation page size must be a positive integer")
        if resolved > self.lifecycle_limits.operation_page_size:
            raise ValueError("operation page size exceeds configured lifecycle limit")
        return resolved

    def list_page(
        self,
        cursor: OperationCursor | None = None,
        *,
        page_size: int | None = None,
    ) -> OperationPage:
        """Return one bounded page using only ``page_size + 1`` ID slots.

        Directory membership, cursors, and raw evidence remain opaque here. The
        lifecycle layer authenticates the bytes before assigning any authority.
        """
        if cursor is not None and not isinstance(cursor, OperationCursor):
            raise TypeError("operation cursor must be an OperationCursor or None")
        limit = self._page_size(page_size)
        operations_directory = resolve_managed_locator(
            self.file_ops.root,
            "operations",
            operation="list_operation_records",
            allow_missing_leaf=True,
        )

        def operation_ids() -> Iterator[str]:
            for path in operations_directory.iterdir():
                name = path.name
                if not name.endswith(".json"):
                    continue
                operation_id = name.removesuffix(".json")
                if cursor is not None and operation_id <= cursor.operation_id:
                    continue
                if len(operation_id) != 32 or any(
                    character not in "0123456789abcdef"
                    for character in operation_id
                ):
                    # Page/checkpoint control evidence lives beside operation
                    # records but must not consume bounded recovery admission.
                    continue
                try:
                    validate_blob_id(operation_id)
                except CacheUnsafePathError:
                    # Hostile names are never evidence or deletion targets.
                    continue
                yield operation_id

        try:
            selected = nsmallest(limit + 1, operation_ids())
        except FileNotFoundError:
            return OperationPage(entries=(), next_cursor=None)
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence directory could not be listed",
                context={"operation": "list_page"},
            ) from exc

        has_more = len(selected) > limit
        page_ids = selected[:limit]
        entries: list[tuple[str, bytes]] = []
        for operation_id in page_ids:
            raw = self._read_bounded(
                self.locator_for(operation_id), operation="list_page"
            )
            if raw is not None:
                entries.append((operation_id, raw))

        next_cursor = (
            OperationCursor(page_ids[-1]) if has_more and page_ids else None
        )
        return OperationPage(entries=tuple(entries), next_cursor=next_cursor)

    def iter_raw(self) -> Iterator[tuple[str, bytes]]:
        """Compatibility iterator composed from bounded cursor pages."""
        cursor: OperationCursor | None = None
        while True:
            page = self.list_page(cursor)
            yield from page.entries
            if page.next_cursor is None:
                return
            cursor = page.next_cursor


__all__ = [
    "FileOperationRecordRepository",
    "OperationCursor",
    "OperationPage",
    "OperationRecordRepository",
    "PendingControlCursor",
    "PendingControlPage",
    "ReconciliationCheckpointCursor",
    "ReconciliationCheckpointPage",
]
