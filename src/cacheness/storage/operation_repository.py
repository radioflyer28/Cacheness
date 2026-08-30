"""Durable exact-byte persistence and bounded paging for lifecycle evidence."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
from heapq import nsmallest
from pathlib import Path
from threading import RLock
from typing import Protocol

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheUnsafePathError,
)

from .operation_record import LifecycleOperationRecord
from .path_security import ManagedFileOps, resolve_managed_locator, validate_blob_id


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
        # A fixed set of per-operation stripes protects only in-process
        # read-compare-write evidence transitions. It deliberately avoids one
        # global normal-operation lock and keeps lock memory bounded.
        self._conditional_locks = tuple(RLock() for _ in range(32))

    def _conditional_lock_for(self, operation_id: str) -> RLock:
        """Return a bounded in-process lock stripe for one opaque evidence ID."""
        stripe = hashlib.sha256(operation_id.encode("ascii")).digest()[0]
        return self._conditional_locks[stripe % len(self._conditional_locks)]

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

    def clear_target_page_locator(self, operation_id: str, page_id: str) -> Path:
        """Return the exact contained locator for one target page."""
        return self._clear_target_locator(operation_id, page_id, checkpoint=False)

    def clear_target_checkpoint_locator(self, operation_id: str, page_id: str) -> Path:
        """Return the exact contained locator for one target checkpoint."""
        return self._clear_target_locator(operation_id, page_id, checkpoint=True)

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

    def _get_clear_target_raw(
        self, operation_id: str, page_id: str, *, checkpoint: bool
    ) -> bytes | None:
        """Load opaque exact clear evidence without declaring it authoritative."""
        try:
            return self.file_ops.read_bytes(
                self._clear_target_locator(operation_id, page_id, checkpoint=checkpoint)
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CacheBlobBackendError(
                "Clear target evidence could not be read",
                context={
                    "operation_id": operation_id,
                    "page_id": page_id,
                    "operation": "get_clear_target",
                },
            ) from exc

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
            with self._conditional_lock_for(f"{operation_id}:{page_id}"):
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
        try:
            return self.file_ops.read_bytes(self.locator_for(operation_id))
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be read",
                context={"operation_id": operation_id, "operation": "get_raw"},
            ) from exc

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
            with self._conditional_lock_for(record.operation_id):
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
            with self._conditional_lock_for(record.operation_id):
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
            try:
                raw = self.file_ops.read_bytes(self.locator_for(operation_id))
            except FileNotFoundError:
                # Concurrent exact retirement is an idempotent absence.
                continue
            except OSError as exc:
                raise CacheBlobBackendError(
                    "Lifecycle operation evidence could not be read",
                    context={"operation_id": operation_id, "operation": "list_page"},
                ) from exc
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
