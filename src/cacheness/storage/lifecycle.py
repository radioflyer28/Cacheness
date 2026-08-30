"""Single authority coordinator for direct BlobStore generation publication."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cacheness.error_handling import (
    CacheBlobRecoverableCleanupError,
    CacheManifestIntegrityError,
    CacheStorageError,
    CacheUnsafePathError,
)

from .integrity import sha256_and_size, sign_hmac_sha256, verify_hmac_sha256
from .manifest import BlobManifestV1
from .manifest_repository import ManifestExpectation
from .operation_record import (
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

    def __init__(self, store: Any):
        self.store = store
        self.operation_repository = FileOperationRecordRepository(
            store.guarded_handler_io.file_ops
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
        updated = self._signed_record(record.at_checkpoint(checkpoint))
        self.operation_repository.checkpoint(updated, updated.canonical_bytes())
        return updated

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
        current = self.store._load_authenticated_manifest(
            record.key,
            operation="recover_operation",
            require_locator=True,
        )
        if current is None:
            if record.expected_generation is not None:
                return
            self.store._delete_or_prove_absent(candidate_locator)
            self.operation_repository.retire(record)
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
            self.operation_repository.retire(record)
            return

        if (
            record.expected_generation is not None
            and manifest.generation == record.expected_generation
        ):
            self.store._delete_or_prove_absent(candidate_locator)
            self.operation_repository.retire(record)

    def recover(self) -> None:
        """Resume authenticated lifecycle debt during store initialization.

        Normal reads never call this method.  A record that is malformed,
        unauthenticated, from another store, or locator-invalid remains
        untouched rather than becoming authority or a deletion target.
        """
        for operation_id, raw in self.operation_repository.iter_raw():
            recovered = self._recoverable_record(operation_id, raw)
            if recovered is None:
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
            self.operation_repository.create(record, record.canonical_bytes())
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
            self.store.manifest_repository.publish_if_expected(
                key,
                expected,
                signed_manifest.canonical_bytes(),
                entry_data=self.store._manifest_entry_data(signed_manifest),
            )
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
                self.operation_repository.retire(record)
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


__all__ = ["LifecycleEngine"]
