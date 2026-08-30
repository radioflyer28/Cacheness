"""Durable exact-byte persistence for lifecycle operation evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from cacheness.error_handling import CacheBlobBackendError

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


__all__ = ["FileOperationRecordRepository", "OperationRecordRepository"]
