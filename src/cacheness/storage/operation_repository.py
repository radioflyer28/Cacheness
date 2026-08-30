"""Durable exact-byte persistence for lifecycle operation evidence."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from cacheness.error_handling import CacheBlobBackendError, CacheUnsafePathError

from .operation_record import LifecycleOperationRecord
from .path_security import ManagedFileOps, resolve_managed_locator, validate_blob_id


class OperationRecordRepository(Protocol):
    """Exact-byte evidence storage used by lifecycle and recovery code."""

    def create(self, record: LifecycleOperationRecord, raw_record: bytes) -> Path:
        """Durably create evidence before a managed candidate side effect."""

    def get_raw(self, operation_id: str) -> bytes | None:
        """Return exact evidence bytes, or ``None`` only for an absent record."""

    def checkpoint(self, record: LifecycleOperationRecord, raw_record: bytes) -> None:
        """Durably replace an existing operation record at a monotonic checkpoint."""

    def retire(self, record: LifecycleOperationRecord) -> None:
        """Retire proven-complete evidence without affecting payload authority."""

    def iter_raw(self) -> Iterator[tuple[str, bytes]]:
        """Yield exact evidence bytes for recovery without assigning authority."""


class FileOperationRecordRepository:
    """Local operation evidence repository inside the managed store root."""

    def __init__(self, file_ops: ManagedFileOps):
        self.file_ops = file_ops

    def locator_for(self, operation_id: str) -> Path:
        """Derive a contained locator from an opaque operation identifier."""
        safe_operation_id = validate_blob_id(operation_id)
        return resolve_managed_locator(
            self.file_ops.root,
            Path("operations") / f"{safe_operation_id}.json",
            operation="operation_record",
            allow_missing_leaf=True,
        )

    def create(self, record: LifecycleOperationRecord, raw_record: bytes) -> Path:
        """Create evidence exclusively and durably before payload publication."""
        try:
            return self.file_ops.create_bytes_durable_exclusive(
                self.locator_for(record.operation_id), raw_record
            )
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be created",
                context={"operation_id": record.operation_id, "operation": "create"},
            ) from exc

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

    def checkpoint(self, record: LifecycleOperationRecord, raw_record: bytes) -> None:
        """Durably persist one authenticated monotonic progress checkpoint."""
        try:
            self.file_ops.write_bytes_durable(
                self.locator_for(record.operation_id), raw_record
            )
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be checkpointed",
                context={
                    "operation_id": record.operation_id,
                    "operation": "checkpoint",
                },
            ) from exc

    def retire(self, record: LifecycleOperationRecord) -> None:
        """Remove evidence only after the lifecycle proves terminal cleanup."""
        try:
            self.file_ops.delete_durable(self.locator_for(record.operation_id))
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence could not be retired",
                context={"operation_id": record.operation_id, "operation": "retire"},
            ) from exc

    def iter_raw(self) -> Iterator[tuple[str, bytes]]:
        """Yield bounded candidates that still need record/authentication checks.

        Directory membership is never treated as payload ownership.  Recovery
        validates the returned raw record, its signature, its store binding,
        and its locators before taking a destructive action.
        """
        operations_directory = resolve_managed_locator(
            self.file_ops.root,
            "operations",
            operation="list_operation_records",
            allow_missing_leaf=True,
        )
        try:
            names = sorted(path.name for path in operations_directory.iterdir())
        except FileNotFoundError:
            return
        except OSError as exc:
            raise CacheBlobBackendError(
                "Lifecycle operation evidence directory could not be listed",
                context={"operation": "iter_raw"},
            ) from exc

        for name in names:
            if not name.endswith(".json"):
                continue
            operation_id = name.removesuffix(".json")
            try:
                validate_blob_id(operation_id)
            except CacheUnsafePathError:
                # A hostile filename is not evidence and is never a deletion
                # target.  A valid signed record must prove ownership instead.
                continue
            try:
                raw = self.file_ops.read_bytes(self.locator_for(operation_id))
            except FileNotFoundError:
                # A concurrent already-retired record is an idempotent absence.
                continue
            except OSError as exc:
                raise CacheBlobBackendError(
                    "Lifecycle operation evidence could not be read",
                    context={"operation_id": operation_id, "operation": "iter_raw"},
                ) from exc
            yield operation_id, raw


__all__ = ["FileOperationRecordRepository", "OperationRecordRepository"]
