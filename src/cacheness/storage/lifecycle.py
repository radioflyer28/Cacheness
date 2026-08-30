"""Single authority coordinator for direct BlobStore generation publication."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobRecoverableCleanupError,
    CacheManifestIntegrityError,
    CacheStorageError,
    CacheUnsafePathError,
)

from .integrity import sha256_and_size, sign_hmac_sha256, verify_hmac_sha256
from .manifest import BlobManifestV1
from .manifest_repository import ManifestCursor, ManifestExpectation
from .operation_record import (
    ClearTarget,
    ClearTargetCheckpoint,
    ClearTargetPage,
    LifecycleOperationRecord,
    OperationCheckpoint,
    OperationKind,
    OperationTransition,
    OPERATION_RECORD_OWNER,
    store_identity,
)
from .operation_repository import FileOperationRecordRepository
from .path_security import resolve_managed_locator, validate_blob_id


class LifecycleEngine:
    """Coordinate the only direct BlobStore write authority transition.

    Native handler serialization remains private.  Once evidence exists, the
    engine either leaves a resumable pre-authority candidate or advances only
    forward from the successful manifest compare-and-swap authority point.
    """

    def __init__(self, store: Any, *, lifecycle_limits: LifecycleLimits):
        self.store = store
        # This is the caller-owned policy object. Later paging, reconciliation,
        # and close admission consume this same instance rather than copies.
        self.lifecycle_limits = lifecycle_limits
        self.operation_repository = FileOperationRecordRepository(
            store.guarded_handler_io.file_ops,
            lifecycle_limits=lifecycle_limits,
        )
        self.test_hook: Callable[[str, LifecycleOperationRecord], None] | None = None
        self.fault_hook: Callable[[str, LifecycleOperationRecord], None] | None = None
        self.recover()

    def _emit(self, step: str, record: LifecycleOperationRecord) -> None:
        if self.test_hook is not None:
            self.test_hook(step, record)

    def _fault(self, seam: str, record: LifecycleOperationRecord) -> None:
        """Invoke a deterministic test-only interruption seam before I/O."""
        if self.fault_hook is not None:
            self.fault_hook(seam, record)

    def _signed_record(
        self,
        record: LifecycleOperationRecord,
        *,
        initialize_new_store: bool = False,
    ) -> LifecycleOperationRecord:
        return record.with_signature(
            sign_hmac_sha256(
                record.signing_bytes(),
                self.store._manifest_key(initialize_new_store=initialize_new_store),
            )
        )

    def _checkpoint(
        self, record: LifecycleOperationRecord, checkpoint: OperationCheckpoint
    ) -> LifecycleOperationRecord:
        expected_raw = record.canonical_bytes()
        updated = self._signed_record(record.at_checkpoint(checkpoint))
        self.operation_repository.checkpoint_if_exact(
            updated,
            expected_raw=expected_raw,
            raw_record=updated.canonical_bytes(),
        )
        return updated

    def _retire(self, record: LifecycleOperationRecord) -> None:
        """Retire only the exact terminal evidence observed by this lifecycle."""
        self.operation_repository.retire_if_exact(
            record, expected_raw=record.canonical_bytes()
        )

    def _signed_clear_target_page(
        self, page: ClearTargetPage
    ) -> ClearTargetPage:
        """Authenticate one bounded clear inventory before it becomes control data."""
        return page.with_signature(
            sign_hmac_sha256(page.signing_bytes(), self.store._manifest_key())
        )

    def _signed_clear_target_checkpoint(
        self, checkpoint: ClearTargetCheckpoint
    ) -> ClearTargetCheckpoint:
        """Authenticate monotonic clear progress independently of manifest bytes."""
        return checkpoint.with_signature(
            sign_hmac_sha256(checkpoint.signing_bytes(), self.store._manifest_key())
        )

    def _clear_page_id(self, operation_id: str, source_cursor: str | None) -> str:
        """Derive a resumable opaque page identifier from its exact source cursor."""
        source = "" if source_cursor is None else source_cursor
        return hashlib.sha256(
            f"{operation_id}\x00{source}".encode("utf-8")
        ).hexdigest()[:32]

    def _authenticated_clear_target_page(
        self,
        raw_page: bytes,
        *,
        operation_id: str,
        page_id: str,
        source_cursor: str | None,
    ) -> ClearTargetPage:
        """Decode one exact page only after its signature and identity agree."""
        page = ClearTargetPage.from_canonical_bytes(raw_page)
        if page.canonical_bytes() != raw_page:
            raise CacheManifestIntegrityError("Clear target page bytes are not canonical")
        if (
            page.operation_id != operation_id
            or page.page_id != page_id
            or page.source_cursor != source_cursor
        ):
            raise CacheManifestIntegrityError("Clear target page identity is inconsistent")
        if not verify_hmac_sha256(
            page.signing_bytes(), page.signature, self.store._manifest_key()
        ):
            raise CacheManifestIntegrityError("Clear target page signature is invalid")
        return page

    def _authenticated_clear_target_checkpoint(
        self,
        raw_checkpoint: bytes,
        *,
        page: ClearTargetPage,
    ) -> ClearTargetCheckpoint:
        """Decode page-bound progress only after exact control-data authentication."""
        checkpoint = ClearTargetCheckpoint.from_canonical_bytes(raw_checkpoint)
        if checkpoint.canonical_bytes() != raw_checkpoint:
            raise CacheManifestIntegrityError(
                "Clear target checkpoint bytes are not canonical"
            )
        if (
            checkpoint.operation_id != page.operation_id
            or checkpoint.page_id != page.page_id
            or checkpoint.page_record_digest
            != hashlib.sha256(page.canonical_bytes()).hexdigest()
        ):
            raise CacheManifestIntegrityError("Clear target checkpoint is not page-bound")
        if not verify_hmac_sha256(
            checkpoint.signing_bytes(),
            checkpoint.signature,
            self.store._manifest_key(),
        ):
            raise CacheManifestIntegrityError("Clear target checkpoint signature is invalid")
        if any(index >= len(page.targets) for index in checkpoint.completed_target_indices):
            raise CacheManifestIntegrityError("Clear target checkpoint exceeds its page")
        if checkpoint.page_complete and checkpoint.completed_target_indices != tuple(
            range(len(page.targets))
        ):
            raise CacheManifestIntegrityError("Clear target page completion is incomplete")
        return checkpoint

    def _authenticated_clear_target(self, target: ClearTarget) -> None:
        """Validate snapshot manifest authority without deriving a payload path."""
        manifest = BlobManifestV1.from_canonical_bytes(target.raw_record)
        if manifest.canonical_bytes() != target.raw_record:
            raise CacheManifestIntegrityError("Clear target manifest bytes are not canonical")
        if (
            manifest.key != target.key
            or manifest.generation != target.generation
            or manifest.state != "committed"
        ):
            raise CacheManifestIntegrityError("Clear target manifest identity is invalid")
        if not verify_hmac_sha256(
            manifest.signing_bytes(), manifest.signature, self.store._manifest_key()
        ):
            raise CacheManifestIntegrityError("Clear target manifest signature is invalid")
        resolve_managed_locator(
            self.store.guarded_handler_io.root,
            manifest.locator,
            operation="clear_snapshot",
        )

    def _new_clear_record(self) -> LifecycleOperationRecord:
        """Create signed control evidence before a bounded clear snapshot begins."""
        operation_id = uuid.uuid4().hex
        root = self.store.guarded_handler_io.root
        timestamp = datetime.now(timezone.utc).isoformat()
        record = LifecycleOperationRecord(
            schema_version=2,
            operation_id=operation_id,
            kind=OperationKind.CLEAR,
            key="clear",
            owner=OPERATION_RECORD_OWNER,
            store_id=store_identity(str(root)),
            topology={
                "backend": type(self.store.backend).__name__,
                "root": store_identity(str(root)),
            },
            expected_generation=None,
            expected_record_digest=None,
            generation=uuid.uuid4().hex,
            candidate_locator=str(
                self.operation_repository.locator_for(operation_id).relative_to(root)
            ),
            previous_locator=None,
            transition=OperationTransition.CLEAR,
            checkpoint=OperationCheckpoint.PREPARED,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return self._signed_record(record, initialize_new_store=True)

    def _load_or_create_clear_checkpoint(
        self, page: ClearTargetPage
    ) -> tuple[ClearTargetCheckpoint, bytes]:
        """Persist zero progress before a page can reclaim any target payload."""
        raw_checkpoint = self.operation_repository.get_clear_target_checkpoint_raw(
            page.operation_id, page.page_id
        )
        if raw_checkpoint is None:
            checkpoint = self._signed_clear_target_checkpoint(
                ClearTargetCheckpoint.initial_for(page)
            )
            raw_checkpoint = checkpoint.canonical_bytes()
            try:
                self.operation_repository.create_clear_target_checkpoint_exclusive(
                    page.operation_id, page.page_id, raw_checkpoint
                )
            except CacheBlobLifecycleConflictError:
                raw_checkpoint = (
                    self.operation_repository.get_clear_target_checkpoint_raw(
                        page.operation_id, page.page_id
                    )
                )
                if raw_checkpoint is None:
                    raise
        checkpoint = self._authenticated_clear_target_checkpoint(
            raw_checkpoint, page=page
        )
        return checkpoint, raw_checkpoint

    def _snapshot_clear_targets(
        self, record: LifecycleOperationRecord
    ) -> LifecycleOperationRecord:
        """Write every bounded authenticated page before releasing admission."""
        source_cursor: str | None = None
        while True:
            page_id = self._clear_page_id(record.operation_id, source_cursor)
            raw_page = self.operation_repository.get_clear_target_page_raw(
                record.operation_id, page_id
            )
            if raw_page is None:
                page = self.store.manifest_repository.list_page(
                    None if source_cursor is None else ManifestCursor(source_cursor),
                    page_size=self.lifecycle_limits.manifest_page_size,
                )
                if not page.entries:
                    if page.next_cursor is not None:
                        raise CacheManifestIntegrityError(
                            "Clear manifest page has an empty non-terminal cursor"
                        )
                    break
                targets = []
                for key, raw_manifest in page.entries:
                    manifest = BlobManifestV1.from_canonical_bytes(raw_manifest)
                    target = ClearTarget.from_raw(
                        key, manifest.generation, raw_manifest
                    )
                    self._authenticated_clear_target(target)
                    targets.append(target)
                page = self._signed_clear_target_page(
                    ClearTargetPage(
                        operation_id=record.operation_id,
                        page_id=page_id,
                        source_cursor=source_cursor,
                        next_cursor=(
                            None
                            if page.next_cursor is None
                            else page.next_cursor.key
                        ),
                        targets=tuple(targets),
                    )
                )
                raw_page = page.canonical_bytes()
                try:
                    self.operation_repository.create_clear_target_page_exclusive(
                        record.operation_id, page_id, raw_page
                    )
                except CacheBlobLifecycleConflictError:
                    raw_page = self.operation_repository.get_clear_target_page_raw(
                        record.operation_id, page_id
                    )
                    if raw_page is None:
                        raise
                page = self._authenticated_clear_target_page(
                    raw_page,
                    operation_id=record.operation_id,
                    page_id=page_id,
                    source_cursor=source_cursor,
                )
            else:
                page = self._authenticated_clear_target_page(
                    raw_page,
                    operation_id=record.operation_id,
                    page_id=page_id,
                    source_cursor=source_cursor,
                )
            self._load_or_create_clear_checkpoint(page)
            if page.next_cursor is None:
                break
            source_cursor = page.next_cursor
        if record.checkpoint is OperationCheckpoint.PREPARED:
            record = self._advance_to(record, OperationCheckpoint.CANDIDATE_PUBLISHED)
        return record

    def _checkpoint_clear_target(
        self,
        page: ClearTargetPage,
        checkpoint: ClearTargetCheckpoint,
        raw_checkpoint: bytes,
        *,
        index: int | None = None,
        complete_page: bool = False,
    ) -> tuple[ClearTargetCheckpoint, bytes]:
        """CAS-persist one monotonic target/page progress transition."""
        updated = checkpoint
        if index is not None:
            updated = updated.with_completed_target(index)
        if complete_page:
            updated = updated.complete_page(len(page.targets))
        updated = self._signed_clear_target_checkpoint(updated)
        updated_raw = updated.canonical_bytes()
        self.operation_repository.checkpoint_clear_target_if_exact(
            page.operation_id,
            page.page_id,
            expected_raw=raw_checkpoint,
            raw_record=updated_raw,
        )
        return updated, updated_raw

    def _delete_clear_target(self, target: ClearTarget) -> bool:
        """Delete only a still-exact snapshot generation through tombstone flow."""
        raw_manifest = self.store.manifest_repository.get_raw(target.key)
        if raw_manifest is None:
            # Missing authority never authorizes a filename or provenance guess.
            return False
        if raw_manifest != target.raw_record:
            raise CacheBlobLifecycleConflictError(
                "BlobStore clear target authority changed after snapshot",
                context={"key": target.key, "operation": "clear"},
            )
        return self.delete(
            key=target.key,
            expected_raw=target.raw_record,
            expected_generation=target.generation,
        )

    def _continue_clear(self, record: LifecycleOperationRecord) -> int:
        """Reclaim exact snapshot targets after aggregate admission is released."""
        if record.checkpoint is OperationCheckpoint.CANDIDATE_PUBLISHED:
            record = self._advance_to(record, OperationCheckpoint.AUTHORITY_PUBLISHED)
        if record.checkpoint is OperationCheckpoint.AUTHORITY_PUBLISHED:
            record = self._advance_to(record, OperationCheckpoint.RECLAIMING)

        cleared = 0
        source_cursor: str | None = None
        while True:
            page_id = self._clear_page_id(record.operation_id, source_cursor)
            raw_page = self.operation_repository.get_clear_target_page_raw(
                record.operation_id, page_id
            )
            if raw_page is None:
                if source_cursor is None:
                    break
                raise CacheManifestIntegrityError("Clear target page is missing")
            page = self._authenticated_clear_target_page(
                raw_page,
                operation_id=record.operation_id,
                page_id=page_id,
                source_cursor=source_cursor,
            )
            checkpoint, raw_checkpoint = self._load_or_create_clear_checkpoint(page)
            for index, target in enumerate(page.targets):
                if index in checkpoint.completed_target_indices:
                    continue
                self._fault("clear_target_delete", record)
                try:
                    if self._delete_clear_target(target):
                        cleared += 1
                except CacheBlobLifecycleConflictError:
                    # A newer manifest generation is outside this clear target.
                    self._emit("clear_target_conflicted", record)
                checkpoint, raw_checkpoint = self._checkpoint_clear_target(
                    page,
                    checkpoint,
                    raw_checkpoint,
                    index=index,
                )
                self._fault("clear_target_checkpoint", record)
            if not checkpoint.page_complete:
                checkpoint, raw_checkpoint = self._checkpoint_clear_target(
                    page,
                    checkpoint,
                    raw_checkpoint,
                    complete_page=True,
                )
            if page.next_cursor is None:
                break
            source_cursor = page.next_cursor
        if record.checkpoint is not OperationCheckpoint.TERMINAL:
            record = self._advance_to(record, OperationCheckpoint.TERMINAL)
        self._retire(record)
        return cleared

    def _recover_clear(self, record: LifecycleOperationRecord) -> None:
        """Resume a retained clear from authenticated snapshot/progress evidence."""
        if record.checkpoint is OperationCheckpoint.PREPARED:
            with self.store._admission_barrier.aggregate_admission():
                record = self._snapshot_clear_targets(record)
        if record.checkpoint is not OperationCheckpoint.PREPARED:
            self._continue_clear(record)

    def clear(self) -> int:
        """Clear one authenticated finite target snapshot through tombstone deletion."""
        record = self._new_clear_record()
        with self.store._admission_barrier.aggregate_admission():
            self.operation_repository.create_exclusive(record, record.canonical_bytes())
            record = self._snapshot_clear_targets(record)
            self._fault("clear_snapshot_complete", record)
        return self._continue_clear(record)

    def is_pre_authority_candidate_eligible(
        self,
        record: LifecycleOperationRecord,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Return whether already-authenticated pre-authority debt passed grace.

        Callers must first authenticate and topology-bind ``record`` through
        ``_recoverable_record``. Age alone never supplies recovery authority.
        """
        if record.checkpoint not in {
            OperationCheckpoint.PREPARED,
            OperationCheckpoint.CANDIDATE_PUBLISHED,
        }:
            return False
        observed_at = datetime.now(timezone.utc) if now is None else now
        if observed_at.tzinfo is None:
            raise ValueError("orphan grace checks require a timezone-aware clock")
        checkpoint_at = datetime.fromisoformat(record.updated_at)
        return observed_at >= checkpoint_at + timedelta(
            seconds=self.lifecycle_limits.orphan_grace_seconds
        )

    def _advance_to(
        self,
        record: LifecycleOperationRecord,
        checkpoint: OperationCheckpoint,
    ) -> LifecycleOperationRecord:
        """Persist each legal monotonic checkpoint through one target state."""
        sequence = (
            OperationCheckpoint.PREPARED,
            OperationCheckpoint.CANDIDATE_PUBLISHED,
            OperationCheckpoint.AUTHORITY_PUBLISHED,
            OperationCheckpoint.RECLAIMING,
            OperationCheckpoint.TERMINAL,
        )
        current_index = sequence.index(record.checkpoint)
        target_index = sequence.index(checkpoint)
        for next_checkpoint in sequence[current_index + 1 : target_index + 1]:
            record = self._checkpoint(record, next_checkpoint)
        return record

    def _recoverable_record(
        self,
        operation_id: str,
        raw: bytes,
    ) -> tuple[LifecycleOperationRecord, Path, Path | None] | None:
        """Authenticate and validate one record before any recovery mutation."""
        try:
            record = LifecycleOperationRecord.from_canonical_bytes(raw)
        except CacheManifestIntegrityError:
            return None
        if record.operation_id != operation_id:
            return None
        if record.canonical_bytes() != raw:
            return None
        if record.store_id != store_identity(str(self.store.guarded_handler_io.root)):
            return None
        if record.owner != OPERATION_RECORD_OWNER:
            return None
        if record.topology != {
            "backend": type(self.store.backend).__name__,
            "root": store_identity(str(self.store.guarded_handler_io.root)),
        }:
            return None
        if not verify_hmac_sha256(
            record.signing_bytes(),
            record.signature,
            self.store._manifest_key(),
        ):
            return None
        try:
            candidate_locator = resolve_managed_locator(
                self.store.guarded_handler_io.root,
                record.candidate_locator,
                operation="recover_operation_candidate",
            )
            previous_locator = (
                None
                if record.previous_locator is None
                else resolve_managed_locator(
                    self.store.guarded_handler_io.root,
                    record.previous_locator,
                    operation="recover_operation_previous",
                )
            )
        except CacheUnsafePathError:
            return None
        return record, candidate_locator, previous_locator

    def _recover_record(
        self,
        record: LifecycleOperationRecord,
        candidate_locator: Path,
        previous_locator: Path | None,
    ) -> None:
        """Converge one authenticated record without guessing manifest authority."""
        if record.transition is OperationTransition.CLEAR:
            self._recover_clear(record)
            return
        if record.transition is OperationTransition.TOMBSTONE:
            self._recover_tombstone(record, candidate_locator)
            return
        current = self.store._load_authenticated_manifest(
            record.key,
            operation="recover_operation",
            require_locator=True,
        )
        if current is None:
            if record.expected_generation is not None:
                return
            if not self.is_pre_authority_candidate_eligible(record):
                return
            self.store._delete_or_prove_absent(candidate_locator)
            self._retire(record)
            return

        manifest, _, current_locator = current
        assert current_locator is not None
        if manifest.generation == record.generation:
            if current_locator != candidate_locator:
                # The same generation must retain the exact operation-owned
                # locator; deleting either side would be speculative.
                return
            record = self._advance_to(record, OperationCheckpoint.AUTHORITY_PUBLISHED)
            if record.checkpoint != OperationCheckpoint.TERMINAL:
                record = self._advance_to(record, OperationCheckpoint.RECLAIMING)
            if previous_locator is not None and previous_locator != candidate_locator:
                self.store._delete_or_prove_absent(previous_locator)
            if record.checkpoint != OperationCheckpoint.TERMINAL:
                record = self._advance_to(record, OperationCheckpoint.TERMINAL)
            self._retire(record)
            return

        if (
            record.expected_generation is not None
            and manifest.generation == record.expected_generation
        ):
            self.store._delete_or_prove_absent(candidate_locator)
            self._retire(record)

    def _tombstone_expectation(
        self,
        key: str,
        manifest: BlobManifestV1,
    ) -> ManifestExpectation | None:
        """Return an exact expectation only for still-current authenticated bytes."""
        raw_manifest = self.store.manifest_repository.get_raw(key)
        if raw_manifest != manifest.canonical_bytes():
            return None
        return ManifestExpectation.from_authenticated_record(
            manifest.generation,
            raw_manifest,
        )

    def _recover_tombstone(
        self,
        record: LifecycleOperationRecord,
        payload_locator: Path,
    ) -> None:
        """Resume only the signed tombstone created by this operation."""
        current = self.store._load_authenticated_manifest(
            record.key,
            operation="recover_delete",
            require_locator=True,
            allowed_states=frozenset({"committed", "tombstoned"}),
        )
        if current is None:
            # A missing authoritative record does not prove that the old
            # payload is still ours to delete. Preserve evidence for later
            # reconciliation instead of inferring ownership from a path.
            return

        manifest, _handler, current_locator = current
        assert current_locator is not None
        if manifest.state == "committed":
            # The tombstone never became authority (or a newer writer won).
            # This delete has no candidate bytes of its own to reclaim.
            self._retire(record)
            return
        if (
            manifest.generation != record.generation
            or current_locator != payload_locator
        ):
            return

        expectation = self._tombstone_expectation(record.key, manifest)
        if expectation is None:
            return
        record = self._advance_to(record, OperationCheckpoint.AUTHORITY_PUBLISHED)
        record = self._advance_to(record, OperationCheckpoint.RECLAIMING)
        self.store._delete_or_prove_absent(payload_locator)
        record = self._advance_to(record, OperationCheckpoint.TERMINAL)
        try:
            self.store.manifest_repository.remove_if_expected(record.key, expectation)
        except CacheBlobLifecycleConflictError:
            # A later generation survived a stale tombstone finalizer. The old
            # payload is already reclaimed, so the exact operation evidence
            # can safely retire without touching the new authority.
            self._retire(record)
            return
        self._retire(record)

    def _find_tombstone_record(
        self,
        key: str,
        generation: str,
    ) -> tuple[LifecycleOperationRecord, Path] | None:
        """Find one authenticated delete record for an observed tombstone."""
        cursor = None
        remaining_actions = self.lifecycle_limits.max_reconcile_actions
        while remaining_actions > 0:
            page = self.operation_repository.list_page(
                cursor,
                page_size=min(
                    self.lifecycle_limits.operation_page_size,
                    remaining_actions,
                ),
            )
            for operation_id, raw in page.entries:
                recovered = self._recoverable_record(operation_id, raw)
                remaining_actions -= 1
                if recovered is None:
                    continue
                record, candidate_locator, _previous_locator = recovered
                if (
                    record.transition is OperationTransition.TOMBSTONE
                    and record.key == key
                    and record.generation == generation
                ):
                    return record, candidate_locator
                if remaining_actions == 0:
                    return None
            if page.next_cursor is None:
                return None
            cursor = page.next_cursor
        return None

    def recover(self) -> None:
        """Resume authenticated lifecycle debt during store initialization.

        Normal reads never call this method.  A record that is malformed,
        unauthenticated, from another store, or locator-invalid remains
        untouched rather than becoming authority or a deletion target.
        """
        cursor = None
        remaining_actions = self.lifecycle_limits.max_reconcile_actions
        while remaining_actions > 0:
            page = self.operation_repository.list_page(
                cursor,
                page_size=min(
                    self.lifecycle_limits.operation_page_size,
                    remaining_actions,
                ),
            )
            for operation_id, raw in page.entries:
                recovered = self._recoverable_record(operation_id, raw)
                if recovered is None:
                    remaining_actions -= 1
                    if remaining_actions == 0:
                        return
                    continue
                record, candidate_locator, previous_locator = recovered
                try:
                    self._recover_record(record, candidate_locator, previous_locator)
                except (CacheStorageError, OSError) as exc:
                    raise CacheBlobRecoverableCleanupError(
                        "BlobStore lifecycle recovery needs another cleanup attempt",
                        context={
                            "operation_id": record.operation_id,
                            "generation": record.generation,
                            "key": record.key,
                        },
                    ) from exc
                remaining_actions -= 1
                if remaining_actions == 0:
                    return
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    def put(
        self,
        data: Any,
        *,
        key: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Privately serialize, conditionally publish, and clean one generation."""
        handler = self.store.handlers.get_handler(data)
        with self.store.guarded_handler_io.stage(handler, data, self.store.config) as staged:
            existing = self.store._load_authenticated_manifest(
                key,
                operation="overwrite",
                require_locator=True,
            )
            previous_manifest = None if existing is None else existing[0]
            previous_locator = None if existing is None else existing[2]
            raw_expected = self.store.manifest_repository.get_raw(key)
            if existing is None:
                expected = ManifestExpectation.absent()
                if raw_expected is not None:
                    # A record appeared after the initial authenticated absence.
                    expected = ManifestExpectation("changed", "changed")
            else:
                assert raw_expected is not None
                assert previous_manifest is not None
                expected = ManifestExpectation.from_authenticated_record(
                    previous_manifest.generation,
                    raw_expected,
                )

            operation_id = uuid.uuid4().hex
            generation = uuid.uuid4().hex
            storage_id = self.store._storage_id_for_key(key)
            candidate_id = validate_blob_id(
                f"{storage_id}-generation-{operation_id}-{generation}{staged.suffix}"
            )
            candidate_locator = self.store.guarded_handler_io.file_ops.blob_locator(
                candidate_id,
                shard_chars=0,
            )
            root = self.store.guarded_handler_io.root
            timestamp = datetime.now(timezone.utc).isoformat()
            record = LifecycleOperationRecord(
                schema_version=2,
                operation_id=operation_id,
                kind=OperationKind.PUT,
                key=key,
                owner=OPERATION_RECORD_OWNER,
                store_id=store_identity(str(self.store.guarded_handler_io.root)),
                topology={
                    "backend": type(self.store.backend).__name__,
                    "root": store_identity(str(root)),
                },
                expected_generation=(
                    None if previous_manifest is None else previous_manifest.generation
                ),
                expected_record_digest=(
                    None
                    if raw_expected is None
                    else hashlib.sha256(raw_expected).hexdigest()
                ),
                generation=generation,
                candidate_locator=str(candidate_locator.relative_to(root)),
                previous_locator=(
                    None
                    if previous_locator is None
                    else str(previous_locator.relative_to(root))
                ),
                transition=(
                    OperationTransition.CREATE
                    if previous_manifest is None
                    else OperationTransition.REPLACE
                ),
                checkpoint=OperationCheckpoint.PREPARED,
                created_at=timestamp,
                updated_at=timestamp,
            )
            record = self._signed_record(
                record,
                initialize_new_store=previous_manifest is None,
            )
            self._fault("evidence_create", record)
            self.operation_repository.create_exclusive(record, record.canonical_bytes())
            self._emit("evidence_created", record)

            self._fault("candidate_publish", record)
            result = self.store.guarded_handler_io.publish_generation(
                staged, candidate_locator
            )
            record = self._checkpoint(record, OperationCheckpoint.CANDIDATE_PUBLISHED)
            self._emit("candidate_published", record)

            self._fault("candidate_verification", record)
            published_locator = Path(result["actual_path"])
            digest, byte_size = sha256_and_size(published_locator)
            if published_locator != candidate_locator:
                raise RuntimeError("Immutable generation publication changed its locator")
            self._emit("candidate_verified", record)

            handler_metadata = dict(result.get("metadata", {}) or {})
            storage_format = result.get("storage_format", "pickle")
            payload_format = result.get(
                "payload_format", getattr(handler, "payload_format", storage_format)
            )
            payload_format_version = result.get(
                "payload_format_version",
                getattr(handler, "payload_format_version", 1),
            )
            handler_metadata["storage_format"] = storage_format
            handler_metadata.setdefault("compression_codec", self.store.compression)
            manifest = BlobManifestV1(
                schema_version=1,
                key=key,
                generation=generation,
                state="committed",
                locator=str(candidate_locator),
                handler_type=handler.data_type,
                payload_format=payload_format,
                payload_format_version=payload_format_version,
                digest_algorithm="sha256",
                digest=digest,
                byte_size=byte_size,
                created_at=datetime.now(timezone.utc).isoformat(),
                handler_metadata=handler_metadata,
                user_metadata=dict(metadata or {}),
            )
            signed_manifest = manifest.with_signature(
                sign_hmac_sha256(
                    manifest.signing_bytes(),
                    self.store._manifest_key(
                        initialize_new_store=previous_manifest is None
                    ),
                )
            )
            self._fault("manifest_publish", record)
            try:
                self.store.manifest_repository.publish_if_expected(
                    key,
                    expected,
                    signed_manifest.canonical_bytes(),
                    entry_data=self.store._manifest_entry_data(signed_manifest),
                )
            except CacheBlobLifecycleConflictError:
                # A failed CAS proves this operation did not become authority.
                # Its immutable locator is evidence-bound to this operation, so
                # it is the only residue this stale contender may reclaim.
                try:
                    self.store._delete_or_prove_absent(candidate_locator)
                    self._retire(record)
                except (CacheStorageError, OSError) as cleanup_error:
                    raise CacheBlobRecoverableCleanupError(
                        "BlobStore stale contender needs candidate cleanup recovery",
                        context={
                            "operation_id": record.operation_id,
                            "generation": record.generation,
                            "key": record.key,
                        },
                    ) from cleanup_error
                raise
            self._fault("authority_checkpoint", record)
            record = self._checkpoint(record, OperationCheckpoint.AUTHORITY_PUBLISHED)
            self._emit("authority_published", record)

            try:
                record = self._checkpoint(record, OperationCheckpoint.RECLAIMING)
                if previous_locator is not None and previous_locator != candidate_locator:
                    self._fault("payload_cleanup", record)
                    self.store._delete_or_prove_absent(previous_locator)
                record = self._checkpoint(record, OperationCheckpoint.TERMINAL)
                self._emit("cleanup_completed", record)
                self._fault("evidence_retire", record)
                self._retire(record)
                self._emit("evidence_retired", record)
            except Exception as exc:
                raise CacheBlobRecoverableCleanupError(
                    "BlobStore authority was published but cleanup needs recovery",
                    context={
                        "operation_id": record.operation_id,
                        "generation": record.generation,
                        "key": record.key,
                    },
                ) from exc
        return key

    def delete(
        self,
        *,
        key: str,
        expected_raw: bytes | None = None,
        expected_generation: str | None = None,
    ) -> bool:
        """Publish signed deletion intent before reclaiming one payload generation."""
        if (expected_raw is None) != (expected_generation is None):
            raise ValueError("Clear deletion expectations require bytes and generation")
        current = self.store._load_authenticated_manifest(
            key,
            operation="delete",
            require_locator=True,
            allowed_states=frozenset({"committed", "tombstoned"}),
        )
        if current is None:
            return False
        manifest, _handler, payload_locator = current
        assert payload_locator is not None

        if expected_raw is not None:
            observed_raw = self.store.manifest_repository.get_raw(key)
            if (
                observed_raw != expected_raw
                or manifest.generation != expected_generation
            ):
                raise CacheBlobLifecycleConflictError(
                    "BlobStore clear target authority changed before tombstone publication",
                    context={"key": key, "operation": "clear"},
                )

        if manifest.state == "tombstoned":
            existing_record = self._find_tombstone_record(key, manifest.generation)
            if existing_record is None:
                raise CacheBlobLifecycleConflictError(
                    "BlobStore tombstone has no matching authenticated operation",
                    context={"key": key, "operation": "delete"},
                )
            record, owned_payload = existing_record
            self._recover_tombstone(record, owned_payload)
            return True

        observed_raw = self.store.manifest_repository.get_raw(key)
        if observed_raw != manifest.canonical_bytes():
            raise CacheBlobLifecycleConflictError(
                "BlobStore delete authority changed before tombstone publication",
                context={"key": key, "operation": "delete"},
            )
        expected = ManifestExpectation.from_authenticated_record(
            manifest.generation,
            observed_raw,
        )
        operation_id = uuid.uuid4().hex
        tombstone_generation = uuid.uuid4().hex
        root = self.store.guarded_handler_io.root
        timestamp = datetime.now(timezone.utc).isoformat()
        record = LifecycleOperationRecord(
            schema_version=2,
            operation_id=operation_id,
            kind=OperationKind.DELETE,
            key=key,
            owner=OPERATION_RECORD_OWNER,
            store_id=store_identity(str(root)),
            topology={
                "backend": type(self.store.backend).__name__,
                "root": store_identity(str(root)),
            },
            expected_generation=manifest.generation,
            expected_record_digest=hashlib.sha256(observed_raw).hexdigest(),
            generation=tombstone_generation,
            candidate_locator=str(payload_locator.relative_to(root)),
            previous_locator=None,
            transition=OperationTransition.TOMBSTONE,
            checkpoint=OperationCheckpoint.PREPARED,
            created_at=timestamp,
            updated_at=timestamp,
        )
        record = self._signed_record(record)
        self._fault("evidence_create", record)
        self.operation_repository.create_exclusive(record, record.canonical_bytes())
        self._emit("evidence_created", record)

        tombstone = replace(
            manifest,
            generation=tombstone_generation,
            state="tombstoned",
            signature="",
        )
        signed_tombstone = tombstone.with_signature(
            sign_hmac_sha256(tombstone.signing_bytes(), self.store._manifest_key())
        )
        tombstone_expectation = ManifestExpectation.from_authenticated_record(
            signed_tombstone.generation,
            signed_tombstone.canonical_bytes(),
        )
        self._fault("tombstone_publish", record)
        try:
            self.store.manifest_repository.publish_if_expected(
                key,
                expected,
                signed_tombstone.canonical_bytes(),
                entry_data=self.store._manifest_entry_data(signed_tombstone),
            )
        except CacheBlobLifecycleConflictError:
            self._retire(record)
            raise
        self._fault("authority_checkpoint", record)
        record = self._checkpoint(record, OperationCheckpoint.CANDIDATE_PUBLISHED)
        record = self._checkpoint(record, OperationCheckpoint.AUTHORITY_PUBLISHED)
        self._emit("authority_published", record)

        try:
            record = self._advance_to(record, OperationCheckpoint.RECLAIMING)
            self._fault("payload_cleanup", record)
            self.store._delete_or_prove_absent(payload_locator)
            record = self._advance_to(record, OperationCheckpoint.TERMINAL)
            self._fault("tombstone_retire", record)
            self.store.manifest_repository.remove_if_expected(key, tombstone_expectation)
            self._retire(record)
            self._emit("evidence_retired", record)
        except CacheBlobLifecycleConflictError:
            # A later generation won after tombstone authority. Never retry its
            # removal; the payload for this exact tombstone was already safe to
            # reclaim and only our terminal evidence is retired.
            self._retire(record)
        except Exception as exc:
            raise CacheBlobRecoverableCleanupError(
                "BlobStore tombstone authority was published but cleanup needs recovery",
                context={
                    "operation_id": record.operation_id,
                    "generation": record.generation,
                    "key": record.key,
                },
            ) from exc
        return True


__all__ = ["LifecycleEngine"]
