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
    CacheBlobReconciliationCheckpointError,
    CacheBlobRecoverableCleanupError,
    CacheManifestIntegrityError,
    CacheStorageError,
    CacheUnsafePathError,
)

from .integrity import sign_hmac_sha256, verify_hmac_sha256
from .manifest import MAX_MANIFEST_BYTES, BlobManifestV1
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
        if self.store._legacy_identity is not None:
            return
        # Opening a second process must not interpret and advance a clear
        # record while its creator still owns the snapshot boundary.
        with self.store._admission_barrier.aggregate_admission():
            self.recover(_snapshot_admitted=True)

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
                record.signing_bytes(lifecycle_limits=self.lifecycle_limits),
                self.store._manifest_key(initialize_new_store=initialize_new_store),
            )
        )

    def _record_raw(self, record: LifecycleOperationRecord) -> bytes:
        """Encode one record through the caller-owned evidence policy."""
        return record.canonical_bytes(lifecycle_limits=self.lifecycle_limits)

    def _checkpoint(
        self, record: LifecycleOperationRecord, checkpoint: OperationCheckpoint
    ) -> LifecycleOperationRecord:
        expected_raw = self._record_raw(record)
        updated = self._signed_record(record.at_checkpoint(checkpoint))
        self.operation_repository.checkpoint_if_exact(
            updated,
            expected_raw=expected_raw,
            raw_record=self._record_raw(updated),
        )
        return updated

    def _retire(self, record: LifecycleOperationRecord) -> None:
        """Retire only the exact terminal evidence observed by this lifecycle."""
        self.operation_repository.retire_if_exact(
            record, expected_raw=self._record_raw(record)
        )

    def _retire_after_conflict(
        self,
        record: LifecycleOperationRecord,
        conflict: CacheBlobLifecycleConflictError,
        *,
        operation: str,
    ) -> None:
        """Retire exact loser evidence without masking a failed cleanup.

        The winner-preserving CAS conflict remains the causal event.  If its
        loser evidence cannot be retired, surface recoverable cleanup debt
        rather than leaking an untyped retirement failure or pretending the
        terminal state was completely converged.
        """
        try:
            self._retire(record)
        except (CacheStorageError, OSError) as cleanup_error:
            raise CacheBlobRecoverableCleanupError(
                "BlobStore conflict left lifecycle evidence requiring cleanup",
                context={
                    "operation_id": record.operation_id,
                    "generation": record.generation,
                    "key": record.key,
                    "operation": operation,
                    "conflict_reason": conflict.context.get("reason"),
                    "conflict_type": type(conflict).__name__,
                    "post_authority": True,
                    "later_winner_preserved": True,
                    "retirement_error": type(cleanup_error).__name__,
                },
            ) from conflict

    def _signed_clear_target_page(
        self, page: ClearTargetPage
    ) -> ClearTargetPage:
        """Authenticate one bounded clear inventory before it becomes control data."""
        return page.with_signature(
            sign_hmac_sha256(
                page.signing_bytes(lifecycle_limits=self.lifecycle_limits),
                self.store._manifest_key(),
            )
        )

    def _signed_clear_target_checkpoint(
        self, checkpoint: ClearTargetCheckpoint
    ) -> ClearTargetCheckpoint:
        """Authenticate monotonic clear progress independently of manifest bytes."""
        return checkpoint.with_signature(
            sign_hmac_sha256(
                checkpoint.signing_bytes(lifecycle_limits=self.lifecycle_limits),
                self.store._manifest_key(),
            )
        )

    def _clear_page_raw(self, page: ClearTargetPage) -> bytes:
        """Encode bounded clear inventory through the same caller policy."""
        return page.canonical_bytes(lifecycle_limits=self.lifecycle_limits)

    def _clear_checkpoint_raw(self, checkpoint: ClearTargetCheckpoint) -> bytes:
        """Encode bounded clear progress through the same caller policy."""
        return checkpoint.canonical_bytes(lifecycle_limits=self.lifecycle_limits)

    def _clear_page_id(
        self,
        operation_id: str,
        source_cursor: ManifestCursor | str | None,
    ) -> str:
        """Derive a resumable opaque identifier from an exact snapshot position.

        Legacy v1 pages only carried a key, so their identifiers retain the
        original preimage. New high-water cursors add a domain marker and both
        sequence components; a same-key event cannot be mistaken for an older
        page after a delete-and-reinsert cycle.
        """
        if source_cursor is None:
            source = ""
            high_water = None
            next_sequence = None
        elif isinstance(source_cursor, str):
            source = source_cursor
            high_water = None
            next_sequence = None
        else:
            source = source_cursor.key
            high_water = source_cursor.snapshot_high_water
            next_sequence = source_cursor.next_sequence
        if high_water is None and next_sequence is None:
            preimage = f"{operation_id}\x00{source}"
        elif high_water is not None and next_sequence is not None:
            preimage = (
                f"{operation_id}\x00clear-snapshot-v2\x00{source}\x00"
                f"{high_water}\x00{next_sequence}"
            )
        else:
            raise CacheManifestIntegrityError("Clear page cursor state is incomplete")
        return hashlib.sha256(
            preimage.encode("utf-8")
        ).hexdigest()[:32]

    @staticmethod
    def _clear_page_cursor(
        key: str | None,
        snapshot_high_water: int | None,
        next_sequence: int | None,
    ) -> ManifestCursor | None:
        """Recover one manifest position from signed clear-page control data."""
        if key is None:
            if snapshot_high_water is not None or next_sequence is not None:
                raise CacheManifestIntegrityError(
                    "Terminal clear page cannot carry a snapshot cursor"
                )
            return None
        try:
            return ManifestCursor(key, snapshot_high_water, next_sequence)
        except ValueError as exc:
            raise CacheManifestIntegrityError(
                "Clear page snapshot cursor is invalid"
            ) from exc

    @staticmethod
    def _clear_page_components(
        cursor: ManifestCursor | None,
    ) -> tuple[str | None, int | None, int | None]:
        """Project a manifest cursor onto the versioned clear-page schema."""
        if cursor is None:
            return None, None, None
        return cursor.key, cursor.snapshot_high_water, cursor.next_sequence

    def _new_clear_target_page(
        self,
        *,
        operation_id: str,
        source_cursor: ManifestCursor | None,
        next_cursor: ManifestCursor | None,
        targets: tuple[ClearTarget, ...],
    ) -> ClearTargetPage:
        """Build one v2 page without losing its generation-bound continuations."""
        source_key, source_high_water, source_next_sequence = self._clear_page_components(
            source_cursor
        )
        next_key, next_high_water, next_next_sequence = self._clear_page_components(
            next_cursor
        )
        return ClearTargetPage(
            operation_id=operation_id,
            page_id=self._clear_page_id(operation_id, source_cursor),
            source_cursor=source_key,
            next_cursor=next_key,
            targets=targets,
            source_snapshot_high_water=source_high_water,
            source_next_sequence=source_next_sequence,
            next_snapshot_high_water=next_high_water,
            next_next_sequence=next_next_sequence,
        )

    def _authenticated_clear_target_page(
        self,
        raw_page: bytes,
        *,
        operation_id: str,
        page_id: str,
        source_cursor: ManifestCursor | None,
        resolve_references: bool = True,
        validate_source_cursor: bool = True,
    ) -> ClearTargetPage:
        """Decode one exact page only after its signature and identity agree."""
        page = ClearTargetPage.from_canonical_bytes(
            raw_page, lifecycle_limits=self.lifecycle_limits
        )
        if self._clear_page_raw(page) != raw_page:
            raise CacheManifestIntegrityError("Clear target page bytes are not canonical")
        if (
            page.operation_id != operation_id
            or page.page_id != page_id
            or (
                validate_source_cursor
                and page.source_cursor
                != (None if source_cursor is None else source_cursor.key)
            )
            or (
                validate_source_cursor
                and self._clear_page_cursor(
                    page.source_cursor,
                    page.source_snapshot_high_water,
                    page.source_next_sequence,
                )
                != source_cursor
            )
            or self._clear_page_id(
                page.operation_id,
                self._clear_page_cursor(
                    page.source_cursor,
                    page.source_snapshot_high_water,
                    page.source_next_sequence,
                ),
            )
            != page.page_id
        ):
            raise CacheManifestIntegrityError("Clear target page identity is inconsistent")
        if not verify_hmac_sha256(
            page.signing_bytes(lifecycle_limits=self.lifecycle_limits),
            page.signature,
            self.store._manifest_key(),
        ):
            raise CacheManifestIntegrityError("Clear target page signature is invalid")
        return self._resolve_clear_target_references(page) if resolve_references else page

    def _clear_target_reference_id(
        self, operation_id: str, raw_record: bytes
    ) -> str:
        """Derive a stable opaque sidecar identity from exact manifest bytes."""
        return hashlib.sha256(
            operation_id.encode("ascii") + b"\x00" + raw_record
        ).hexdigest()[:32]

    def _reference_target(
        self,
        *,
        key: str,
        generation: str,
        raw_manifest: bytes,
        operation_id: str,
    ) -> ClearTarget:
        """Build page-bound bounded evidence for one exact oversized manifest."""
        return ClearTarget.from_reference(
            key,
            generation,
            raw_manifest,
            self._clear_target_reference_id(operation_id, raw_manifest),
            chunk_size=self.lifecycle_limits.max_operation_record_bytes,
        )

    def _validate_clear_evidence_contract(
        self, record: LifecycleOperationRecord
    ) -> None:
        """Reject a policy that cannot encode every valid clear control page.

        The primary clear record is deliberately not written until this check
        succeeds.  A page may need to name the largest valid manifest through
        a chunked reference, including every immutable chunk digest required
        for safe read and retirement after a crash.
        """
        self._record_raw(record)
        chunk_size = self.lifecycle_limits.max_operation_record_bytes
        chunk_count = (MAX_MANIFEST_BYTES + chunk_size - 1) // chunk_size
        probe = ClearTarget(
            key="clear-control-probe",
            generation="0" * 32,
            record_digest="0" * 64,
            raw_record=None,
            raw_record_reference="1" * 32,
            raw_record_chunk_count=chunk_count,
            raw_record_byte_length=MAX_MANIFEST_BYTES,
            raw_record_chunk_digests=("0" * 64,) * chunk_count,
        )
        page = ClearTargetPage(
            operation_id=record.operation_id,
            page_id=self._clear_page_id(record.operation_id, None),
            source_cursor=None,
            next_cursor=None,
            targets=(probe,),
        )
        self._clear_page_raw(self._signed_clear_target_page(page))

    def _resolve_clear_target_references(
        self, page: ClearTargetPage
    ) -> ClearTargetPage:
        """Attach page-bound oversized records only after page authentication.

        The signed page binds a sidecar identifier and digest.  The sidecar is
        still treated as untrusted until its exact bytes match that digest, so
        a filename alone never becomes deletion authority.
        """
        resolved: list[ClearTarget] = []
        for target in page.targets:
            if target.raw_record_reference is None:
                resolved.append(target)
                continue
            raw_record = self.operation_repository.get_clear_target_reference_raw(
                page.operation_id,
                target.raw_record_reference,
                chunk_count=target.raw_record_chunk_count,
                byte_length=target.raw_record_byte_length,
                chunk_digests=target.raw_record_chunk_digests,
            )
            if raw_record is None:
                raise CacheManifestIntegrityError("Clear target reference is missing")
            resolved.append(target.with_resolved_raw(raw_record))
        return replace(page, targets=tuple(resolved))

    def _authenticated_clear_target_checkpoint(
        self,
        raw_checkpoint: bytes,
        *,
        page: ClearTargetPage,
    ) -> ClearTargetCheckpoint:
        """Decode page-bound progress only after exact control-data authentication."""
        checkpoint = ClearTargetCheckpoint.from_canonical_bytes(
            raw_checkpoint, lifecycle_limits=self.lifecycle_limits
        )
        if self._clear_checkpoint_raw(checkpoint) != raw_checkpoint:
            raise CacheManifestIntegrityError(
                "Clear target checkpoint bytes are not canonical"
            )
        if (
            checkpoint.operation_id != page.operation_id
            or checkpoint.page_id != page.page_id
            or checkpoint.page_record_digest
            != hashlib.sha256(self._clear_page_raw(page)).hexdigest()
        ):
            raise CacheManifestIntegrityError("Clear target checkpoint is not page-bound")
        if not verify_hmac_sha256(
            checkpoint.signing_bytes(lifecycle_limits=self.lifecycle_limits),
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
            raise CacheBlobLifecycleConflictError(
                "BlobStore clear target is not a committed manifest generation",
                context={"key": target.key, "operation": "clear"},
            )
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
                ClearTargetCheckpoint.initial_for(
                    page, lifecycle_limits=self.lifecycle_limits
                )
            )
            raw_checkpoint = self._clear_checkpoint_raw(checkpoint)
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
        source_cursor: ManifestCursor | None = None
        while True:
            page_id = self._clear_page_id(record.operation_id, source_cursor)
            raw_page = self.operation_repository.get_clear_target_page_raw(
                record.operation_id, page_id
            )
            if raw_page is None:
                manifest_page = self.store.manifest_repository.list_page(
                    source_cursor,
                    page_size=self.lifecycle_limits.manifest_page_size,
                )
                if not manifest_page.entries:
                    # A high-water manifest inventory deliberately retains
                    # membership independently of the current canonical row.
                    # An inspected window can therefore contain only events
                    # subsequently superseded or deleted.  That is a normal
                    # empty *source* page, not corrupted clear authority. Do
                    # not make a zero-target control page; advance the source
                    # cursor and persist the first page that has a target.
                    if manifest_page.next_cursor is None:
                        break
                    source_cursor = manifest_page.next_cursor
                    continue
                if len(manifest_page.entry_next_cursors) != len(manifest_page.entries):
                    raise CacheManifestIntegrityError(
                        "Clear manifest page lacks exact entry continuations"
                    )
                targets: list[ClearTarget] = []
                referenced_manifests: list[tuple[ClearTarget, bytes]] = []
                next_cursor: ManifestCursor | None = None
                for index, (key, raw_manifest) in enumerate(manifest_page.entries):
                    manifest = BlobManifestV1.from_canonical_bytes(raw_manifest)
                    target = ClearTarget.from_raw(
                        key, manifest.generation, raw_manifest
                    )
                    self._authenticated_clear_target(target)
                    # Inventory sequence is an immutable snapshot-membership
                    # order, whereas target pages use an independent canonical
                    # key order for stable per-page checkpoints.  Never infer
                    # source progression from this presentation ordering.
                    candidate_targets = tuple(
                        sorted([*targets, target], key=lambda item: item.key)
                    )
                    has_remaining = (
                        index + 1 < len(manifest_page.entries)
                        or manifest_page.next_cursor is not None
                    )
                    candidate_cursor = (
                        manifest_page.entry_next_cursors[index]
                        if has_remaining
                        else None
                    )
                    candidate = self._new_clear_target_page(
                        operation_id=record.operation_id,
                        source_cursor=source_cursor,
                        next_cursor=candidate_cursor,
                        targets=candidate_targets,
                    )
                    try:
                        # Page limits are encoded-byte limits, not a manifest
                        # count heuristic.  Keep individually valid manifests
                        # in immutable bounded sidecars when needed.
                        self._clear_page_raw(self._signed_clear_target_page(candidate))
                    except CacheManifestIntegrityError:
                        target = self._reference_target(
                            key=key,
                            generation=manifest.generation,
                            raw_manifest=raw_manifest,
                            operation_id=record.operation_id,
                        )
                        candidate_targets = tuple(
                            sorted([*targets, target], key=lambda item: item.key)
                        )
                        candidate = self._new_clear_target_page(
                            operation_id=record.operation_id,
                            source_cursor=source_cursor,
                            next_cursor=candidate_cursor,
                            targets=candidate_targets,
                        )
                        try:
                            self._clear_page_raw(
                                self._signed_clear_target_page(candidate)
                            )
                        except CacheManifestIntegrityError:
                            # The primary clear contract was preflighted before
                            # persistence. A later failure therefore reflects a
                            # malformed/oversized observed manifest, never an
                            # intentionally unrecoverable primary record.
                            if targets:
                                break
                            raise
                        referenced_manifests.append((target, raw_manifest))
                    targets.append(target)
                    next_cursor = candidate_cursor
                if not targets:
                    raise CacheManifestIntegrityError("Clear target page has no bounded target")
                page = self._signed_clear_target_page(
                    self._new_clear_target_page(
                        operation_id=record.operation_id,
                        source_cursor=source_cursor,
                        next_cursor=next_cursor,
                        targets=tuple(sorted(targets, key=lambda item: item.key)),
                    )
                )
                raw_page = self._clear_page_raw(page)
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
                # Persist the signed page before its immutable chunks. A
                # process loss in this pre-authority window can authenticate
                # the page and retire every proven partial chunk without ever
                # treating its targets as deletion authority.
                self._fault("clear_target_page_persisted", record)
                for target, raw_manifest in referenced_manifests:
                    assert target.raw_record_reference is not None
                    self.operation_repository.create_clear_target_reference_chunks_exclusive(
                        record.operation_id,
                        target.raw_record_reference,
                        raw_manifest,
                    )
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
            source_cursor = self._clear_page_cursor(
                page.next_cursor,
                page.next_snapshot_high_water,
                page.next_next_sequence,
            )
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
        updated_raw = self._clear_checkpoint_raw(updated)
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
        """Resume one clear under a dedicated aggregate-control lease.

        The aggregate lease serializes clear progress and recovery but is
        distinct from bounded evidence-CAS stripes. Each target deletion and
        checkpoint obtains its own exact child operation lease, so parent clear
        evidence never grants child CAS authority.
        """
        with self.operation_repository.clear_operation_transition(record.operation_id):
            raw_current = self.operation_repository.get_raw(record.operation_id)
            if raw_current is None:
                return 0
            recovered = self._recoverable_record(record.operation_id, raw_current)
            if recovered is None:
                raise CacheManifestIntegrityError("Clear operation evidence is untrusted")
            current, _candidate, _previous = recovered
            return self._continue_clear_locked(current)

    def _continue_clear_locked(self, record: LifecycleOperationRecord) -> int:
        """Reclaim exact snapshot targets after aggregate admission is released."""
        if record.checkpoint is OperationCheckpoint.TERMINAL:
            self._retire_clear_control_artifacts(record)
            self._retire(record)
            return 0
        if record.checkpoint is OperationCheckpoint.CANDIDATE_PUBLISHED:
            record = self._advance_to(record, OperationCheckpoint.AUTHORITY_PUBLISHED)
        if record.checkpoint is OperationCheckpoint.AUTHORITY_PUBLISHED:
            record = self._advance_to(record, OperationCheckpoint.RECLAIMING)

        cleared = 0
        source_cursor: ManifestCursor | None = None
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
            source_cursor = self._clear_page_cursor(
                page.next_cursor,
                page.next_snapshot_high_water,
                page.next_next_sequence,
            )
        if record.checkpoint is not OperationCheckpoint.TERMINAL:
            record = self._advance_to(record, OperationCheckpoint.TERMINAL)
        self._retire_clear_control_artifacts(record)
        self._retire(record)
        return cleared

    def _retire_clear_control_artifacts(
        self, record: LifecycleOperationRecord
    ) -> None:
        """Retire only completed signed clear artifacts, before the main record.

        Directory names are candidate inventory only.  A page must validate
        against the terminal operation and, when present, its checkpoint must
        prove every target completed.  Retiring checkpoint, referenced bytes,
        and page independently is idempotent; a crash at any point resumes
        from the remaining exact evidence.  The primary operation record is
        intentionally retired last, so no valid clear artifact becomes an
        unowned filename after interruption.
        """
        if record.checkpoint is not OperationCheckpoint.TERMINAL:
            raise CacheManifestIntegrityError("Clear control cleanup requires terminal evidence")
        for page_id in self.operation_repository.iter_clear_target_page_ids(
            record.operation_id
        ):
            raw_page = self.operation_repository.get_clear_target_page_raw(
                record.operation_id, page_id
            )
            if raw_page is None:
                continue
            page = self._authenticated_clear_target_page(
                raw_page,
                operation_id=record.operation_id,
                page_id=page_id,
                source_cursor=None,
                resolve_references=False,
                validate_source_cursor=False,
            )
            raw_checkpoint = self.operation_repository.get_clear_target_checkpoint_raw(
                record.operation_id, page_id
            )
            if raw_checkpoint is not None:
                checkpoint = self._authenticated_clear_target_checkpoint(
                    raw_checkpoint, page=page
                )
                if not checkpoint.page_complete:
                    raise CacheManifestIntegrityError(
                        "Clear target checkpoint is incomplete during retirement"
                    )
                self.operation_repository.retire_clear_target_checkpoint_if_exact(
                    record.operation_id, page_id, expected_raw=raw_checkpoint
                )
            for target in page.targets:
                reference = target.raw_record_reference
                if reference is None:
                    continue
                raw_reference = self.operation_repository.get_clear_target_reference_raw(
                    record.operation_id,
                    reference,
                    chunk_count=target.raw_record_chunk_count,
                    byte_length=target.raw_record_byte_length,
                    chunk_digests=target.raw_record_chunk_digests,
                )
                if raw_reference is None:
                    continue
                # The signed page supplies the reference digest.  Authenticate
                # it before removing the independently persisted bytes.
                target.with_resolved_raw(raw_reference)
                self.operation_repository.retire_clear_target_reference_if_exact(
                    record.operation_id,
                    reference,
                    expected_raw=raw_reference,
                    chunk_count=target.raw_record_chunk_count,
                    byte_length=target.raw_record_byte_length,
                    chunk_digests=target.raw_record_chunk_digests,
                )
            self.operation_repository.retire_clear_target_page_if_exact(
                record.operation_id, page_id, expected_raw=raw_page
            )

    def _abort_prepared_clear(self, record: LifecycleOperationRecord) -> None:
        """Retire only authenticated pre-authority clear control evidence.

        A PREPARED clear has not published its immutable inventory authority.
        It therefore must never resume lexical listing after its admission epoch
        was lost: a key committed by another process after the crash could
        otherwise be added to an old clear. Every page is authenticated before
        its sidecars are removed, and this path deliberately never reads or
        deletes a manifest generation.
        """
        with self.operation_repository.operation_transition(record.operation_id):
            current_raw = self.operation_repository.get_raw(record.operation_id)
            if current_raw is None:
                return
            recovered = self._recoverable_record(record.operation_id, current_raw)
            if recovered is None:
                raise CacheManifestIntegrityError("Clear operation evidence is untrusted")
            current, _candidate, _previous = recovered
            if current.checkpoint is not OperationCheckpoint.PREPARED:
                return
            for page_id in self.operation_repository.iter_clear_target_page_ids(
                current.operation_id
            ):
                raw_page = self.operation_repository.get_clear_target_page_raw(
                    current.operation_id, page_id
                )
                if raw_page is None:
                    continue
                page = self._authenticated_clear_target_page(
                    raw_page,
                    operation_id=current.operation_id,
                    page_id=page_id,
                    source_cursor=None,
                    resolve_references=False,
                    validate_source_cursor=False,
                )
                raw_checkpoint = (
                    self.operation_repository.get_clear_target_checkpoint_raw(
                        current.operation_id, page_id
                    )
                )
                if raw_checkpoint is not None:
                    self._authenticated_clear_target_checkpoint(
                        raw_checkpoint, page=page
                    )
                    self.operation_repository.retire_clear_target_checkpoint_if_exact(
                        current.operation_id, page_id, expected_raw=raw_checkpoint
                    )
                for target in page.targets:
                    reference = target.raw_record_reference
                    if reference is None:
                        continue
                    if target.raw_record_chunk_count is not None:
                        self.operation_repository.retire_clear_target_reference_chunks_if_bound(
                            current.operation_id,
                            reference,
                            chunk_count=target.raw_record_chunk_count,
                            byte_length=target.raw_record_byte_length,
                            chunk_digests=target.raw_record_chunk_digests,
                        )
                    # A legacy one-file reference cannot be safely retired if
                    # it no longer fits the current caller bound. It pre-dates
                    # the signed chunk contract and remains evidence, not a
                    # deletion target.
                    if target.raw_record_chunk_count is None:
                        raw_reference = (
                            self.operation_repository.get_clear_target_reference_raw(
                                current.operation_id, reference
                            )
                        )
                        if raw_reference is not None:
                            target.with_resolved_raw(raw_reference)
                            self.operation_repository.retire_clear_target_reference_if_exact(
                                current.operation_id,
                                reference,
                                expected_raw=raw_reference,
                            )
                self.operation_repository.retire_clear_target_page_if_exact(
                    current.operation_id, page_id, expected_raw=raw_page
                )
            self._retire(current)

    def _recover_clear(
        self, record: LifecycleOperationRecord, *, snapshot_admitted: bool = False
    ) -> None:
        """Resume a retained clear from authenticated snapshot/progress evidence."""
        if record.checkpoint is OperationCheckpoint.PREPARED:
            self._abort_prepared_clear(record)
            return
        if record.checkpoint is not OperationCheckpoint.PREPARED:
            self._continue_clear(record)

    def clear(self) -> int:
        """Clear one authenticated finite target snapshot through tombstone deletion."""
        record = self._new_clear_record()
        self._validate_clear_evidence_contract(record)
        # A live clear owns its continuation lease before it publishes the
        # target snapshot.  Constructor recovery may observe the snapshot as
        # soon as aggregate admission is released, but it must wait for this
        # creator to finish rather than stealing the public return value.
        # Process loss naturally releases the advisory lease, leaving the
        # authenticated record available for a later opener to resume.
        with self.operation_repository.clear_operation_transition(record.operation_id):
            with self.store._admission_barrier.aggregate_admission():
                self.operation_repository.create_exclusive(
                    record, self._record_raw(record)
                )
                record = self._snapshot_clear_targets(record)
                self._fault("clear_snapshot_complete", record)
            self._fault("clear_snapshot_admission_released", record)
            return self._continue_clear_locked(record)

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
            record = LifecycleOperationRecord.from_canonical_bytes(
                raw, lifecycle_limits=self.lifecycle_limits
            )
        except CacheManifestIntegrityError:
            return None
        if record.operation_id != operation_id:
            return None
        if self._record_raw(record) != raw:
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
            record.signing_bytes(lifecycle_limits=self.lifecycle_limits),
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
        *,
        snapshot_admitted: bool = False,
    ) -> None:
        """Converge one authenticated record without guessing manifest authority."""
        if record.transition is OperationTransition.CLEAR:
            self._recover_clear(record, snapshot_admitted=snapshot_admitted)
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
            if not self.store.guarded_handler_io.file_ops.exists(candidate_locator):
                # A signed PREPARED record whose immutable candidate never
                # appeared has no externally persistent payload side effect.
                # It can retire immediately after an interrupted evidence
                # promotion; waiting for orphan grace would retain control
                # residue even though there is no candidate to reconcile.
                self._retire(record)
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
            return

        if (
            manifest.generation != record.generation
            and current_locator != candidate_locator
        ):
            # A different authoritative generation at a different contained
            # immutable locator proves this operation never became authority.
            # The candidate is signed evidence-owned by this exact operation;
            # remove only that residue.  Same-locator/different-generation is
            # deliberately not safe: it may name a later operation's payload.
            if record.checkpoint is OperationCheckpoint.PREPARED:
                record = self._advance_to(
                    record, OperationCheckpoint.CANDIDATE_PUBLISHED
                )
            self.store._delete_or_prove_absent(candidate_locator)
            self._retire(record)

    @staticmethod
    def _tombstone_expectation(
        manifest: BlobManifestV1,
        raw_manifest: bytes,
    ) -> ManifestExpectation:
        """Bind a recovery removal to the exact authenticated snapshot.

        A recovery operation must never perform a second repository read and
        accidentally adopt an intervening winner as its removal expectation.
        """
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
        # Reconciliation owns a tombstone once it has durably written its
        # action checkpoint.  Its state machine retains this primary record
        # until after payload/tombstone progress is checkpointed, so normal
        # startup recovery must not retire the record underneath a prepared
        # sidecar.  Invalid sidecars remain inert evidence and cannot grant
        # deletion authority; reconciliation reports them separately.
        if self._has_authenticated_tombstone_checkpoint(record):
            return
        current = self.store._load_authenticated_manifest_with_raw(
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

        manifest, raw_manifest, _handler, current_locator = current
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

        expectation = self._tombstone_expectation(manifest, raw_manifest)
        record = self._advance_to(record, OperationCheckpoint.AUTHORITY_PUBLISHED)
        record = self._advance_to(record, OperationCheckpoint.RECLAIMING)
        self.store._delete_or_prove_absent(payload_locator)
        record = self._advance_to(record, OperationCheckpoint.TERMINAL)
        try:
            self.store.manifest_repository.remove_if_expected(record.key, expectation)
        except CacheBlobLifecycleConflictError as conflict:
            # A later generation survived a stale tombstone finalizer. The old
            # payload is already reclaimed, so the exact operation evidence
            # can safely retire without touching the new authority.
            self._retire_after_conflict(
                record,
                conflict,
                operation="recover_tombstone_retire_conflict",
            )
            return
        self._retire(record)

    def _has_authenticated_tombstone_checkpoint(
        self, record: LifecycleOperationRecord
    ) -> bool:
        """Return whether a bound sidecar, not its pathname, owns this tombstone."""
        raw = self.operation_repository.get_reconciliation_checkpoint_raw(
            record.operation_id
        )
        if raw is None:
            return False
        primary_raw = self.operation_repository.get_raw(record.operation_id)
        if primary_raw is None:
            return False
        # Import lazily to avoid the lifecycle/reconciliation construction
        # cycle.  A private sidecar can defer startup only when its signature,
        # operation ID, action, and digest all bind this exact primary record.
        from .reconciliation import _ActionCheckpoint, ReconciliationAction

        try:
            checkpoint = _ActionCheckpoint.from_canonical_bytes(
                raw,
                self.store._manifest_key(),
                lifecycle_limits=self.lifecycle_limits,
            )
        except CacheBlobReconciliationCheckpointError:
            return False
        return (
            checkpoint.operation_id == record.operation_id
            and checkpoint.action is ReconciliationAction.COMPLETE_TOMBSTONE
            and checkpoint.evidence_digest == hashlib.sha256(primary_raw).hexdigest()
        )

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

    def recover(self, *, _snapshot_admitted: bool = False) -> None:
        """Resume authenticated lifecycle debt during store initialization.

        Normal reads never call this method.  A record that is malformed,
        unauthenticated, from another store, or locator-invalid remains
        untouched rather than becoming authority or a deletion target.
        """
        self.operation_repository.recover_pending_operation_records()
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
                    self._recover_record(
                        record,
                        candidate_locator,
                        previous_locator,
                        snapshot_admitted=_snapshot_admitted,
                    )
                except CacheBlobRecoverableCleanupError:
                    # Conflict-aware cleanup carries the authoritative winner
                    # and post-authority context.  Do not wrap it into a
                    # generic recovery failure and erase that distinction.
                    raise
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
            existing = self.store._load_authenticated_manifest_with_raw(
                key,
                operation="overwrite",
                require_locator=True,
            )
            previous_manifest = None if existing is None else existing[0]
            raw_expected = None if existing is None else existing[1]
            previous_locator = None if existing is None else existing[3]
            if existing is None:
                expected = ManifestExpectation.absent()
            else:
                # ``raw_expected`` is the exact authenticated authority read
                # above.  Do not reread: an intervening winner must make this
                # operation's CAS conflict rather than becoming its expected
                # record.
                if raw_expected is None or previous_manifest is None:
                    raise CacheBlobLifecycleConflictError(
                        "BlobStore overwrite authority disappeared before expectation",
                        context={"key": key, "operation": "put"},
                    )
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
            self.operation_repository.create_exclusive(record, self._record_raw(record))
            self._emit("evidence_created", record)

            self._fault("candidate_publish", record)
            result = self.store.guarded_handler_io.publish_generation(
                staged, candidate_locator
            )
            record = self._checkpoint(record, OperationCheckpoint.CANDIDATE_PUBLISHED)
            self._emit("candidate_published", record)

            self._fault("candidate_verification", record)
            published_locator = Path(result["actual_path"])
            if published_locator != candidate_locator:
                raise RuntimeError("Immutable generation publication changed its locator")
            published_identity = result.pop("_managed_generation_identity", None)
            if (
                not isinstance(published_identity, tuple)
                or len(published_identity) != 2
                or not all(type(part) is int for part in published_identity)
            ):
                raise RuntimeError("Immutable generation publication did not bind a file identity")
            digest, byte_size = self.store.guarded_handler_io.file_ops.sha256_and_size(
                candidate_locator, expected_identity=published_identity
            )
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
        current = self.store._load_authenticated_manifest_with_raw(
            key,
            operation="delete",
            require_locator=True,
            allowed_states=frozenset({"committed", "tombstoned"}),
        )
        if current is None:
            return False
        manifest, observed_raw, _handler, payload_locator = current
        if payload_locator is None:
            raise CacheBlobLifecycleConflictError(
                "BlobStore delete authority has no managed payload locator",
                context={"key": key, "operation": "delete"},
            )

        if expected_raw is not None:
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
        self.operation_repository.create_exclusive(record, self._record_raw(record))
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
        except CacheBlobLifecycleConflictError as conflict:
            self._retire_after_conflict(
                record, conflict, operation="tombstone_publish_conflict"
            )
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
        except CacheBlobLifecycleConflictError as conflict:
            # A later generation won after tombstone authority. Never retry its
            # removal; the payload for this exact tombstone was already safe to
            # reclaim and only our terminal evidence is retired.
            self._retire_after_conflict(
                record, conflict, operation="tombstone_retire_conflict"
            )
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
