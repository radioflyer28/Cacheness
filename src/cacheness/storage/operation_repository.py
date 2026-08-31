"""Durable exact-byte persistence and bounded paging for lifecycle evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from heapq import nsmallest
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Protocol

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheManifestIntegrityError,
    CacheReason,
    CacheUnsafePathError,
)

from .operation_record import LifecycleOperationRecord
from .path_security import ManagedFileOps, resolve_managed_locator, validate_blob_id


_CONDITIONAL_LOCK_STRIPES = tuple(RLock() for _ in range(64))


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
    def _conditional_lock_for(self, operation_id: str) -> RLock:
        """Return a shared bounded in-process stripe for one evidence ID.

        The stripe is deliberately module-wide so independently constructed
        repositories in the same process cannot bypass their common advisory
        file lock.  It only orders evidence transitions; it is never an
        authority substitute for the cross-process compare-and-mutate lock.
        """
        identity = f"{self.file_ops.root}\x00{operation_id}".encode("utf-8")
        stripe = hashlib.sha256(identity).digest()[0]
        return _CONDITIONAL_LOCK_STRIPES[stripe % len(_CONDITIONAL_LOCK_STRIPES)]

    def _conditional_lock_locator(self, operation_id: str) -> Path:
        """Return one contained fixed advisory lock for an evidence transition."""
        digest = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / ".conditional-locks" / f"{digest}.lock",
            operation="operation_record_conditional_lock",
            allow_missing_leaf=True,
        )

    @contextmanager
    def _conditional_transition(self, operation_id: str) -> Iterator[None]:
        """Serialize one exact evidence CAS across repository objects/processes.

        ``flock`` is acquired only around read/compare/replace-or-delete.  It
        never spans handler serialization, manifest publication, or unrelated
        operation IDs, preserving the phase's no-global-normal-lock contract.
        """
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - non-POSIX topology.
            raise CacheBlobBackendError(
                "Operation evidence requires POSIX advisory locking",
                context={"operation": "conditional_evidence"},
            ) from exc

        lock_locator = self._conditional_lock_locator(operation_id)
        try:
            self.file_ops.create_bytes_durable_exclusive(lock_locator, b"lock\n")
        except FileExistsError:
            pass

        handle: BinaryIO | None = None
        with self._conditional_lock_for(operation_id):
            try:
                handle = self.file_ops.open_read(lock_locator)
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            except OSError as exc:
                raise CacheBlobBackendError(
                    "Operation evidence conditional transition could not lock",
                    context={
                        "operation_id": operation_id,
                        "operation": "conditional_evidence",
                    },
                ) from exc
            finally:
                if handle is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        handle.close()

    @contextmanager
    def operation_transition(self, operation_id: str) -> Iterator[None]:
        """Hold a narrow cross-process lease for one resumable operation."""
        with self._conditional_transition(f"operation-run:{operation_id}"):
            yield

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

    def clear_target_page_locator(self, operation_id: str, page_id: str) -> Path:
        """Return the exact contained locator for one target page."""
        return self._clear_target_locator(operation_id, page_id, checkpoint=False)

    def clear_target_checkpoint_locator(self, operation_id: str, page_id: str) -> Path:
        """Return the exact contained locator for one target checkpoint."""
        return self._clear_target_locator(operation_id, page_id, checkpoint=True)

    def clear_target_reference_locator(self, operation_id: str, reference: str) -> Path:
        """Return the exact contained locator for one referenced manifest copy."""
        return self._clear_target_reference_locator(operation_id, reference)

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

    def get_reconciliation_checkpoint_raw(self, operation_id: str) -> bytes | None:
        """Read opaque reconciliation progress without granting it authority."""
        return self._read_bounded(
            self.reconciliation_checkpoint_locator(operation_id),
            operation="get_reconcile",
        )

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
            with self._conditional_transition(f"reconcile:{operation_id}"):
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
            with self._conditional_transition(f"reconcile:{operation_id}"):
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
        self, operation_id: str, reference: str
    ) -> bytes | None:
        """Read one oversized exact target record without granting authority."""
        return self._read_bounded(
            self._clear_target_reference_locator(operation_id, reference),
            operation="get_clear_target_reference",
        )

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
        with self._conditional_transition(transition_id):
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
        self, operation_id: str, reference: str, *, expected_raw: bytes
    ) -> bool:
        """Retire one authenticated page's oversized exact-manifest sidecar."""
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
            with self._conditional_transition(f"clear:{operation_id}:{page_id}"):
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
            with self._conditional_transition(f"operation:{record.operation_id}"):
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
            with self._conditional_transition(f"operation:{record.operation_id}"):
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
]
