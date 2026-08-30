"""Single authority coordinator for direct BlobStore generation publication."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cacheness.error_handling import CacheBlobRecoverableCleanupError

from .integrity import sha256_and_size, sign_hmac_sha256
from .manifest import BlobManifestV1
from .manifest_repository import ManifestExpectation
from .operation_record import (
    LifecycleOperationRecord,
    OperationCheckpoint,
    OperationKind,
    store_identity,
)
from .operation_repository import FileOperationRecordRepository
from .path_security import validate_blob_id


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

    def _emit(self, step: str, record: LifecycleOperationRecord) -> None:
        if self.test_hook is not None:
            self.test_hook(step, record)

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
            record = LifecycleOperationRecord(
                schema_version=1,
                operation_id=operation_id,
                kind=OperationKind.PUT,
                key=key,
                store_id=store_identity(str(self.store.guarded_handler_io.root)),
                expected_generation=(
                    None if previous_manifest is None else previous_manifest.generation
                ),
                generation=generation,
                candidate_locator=str(candidate_locator),
                previous_locator=(
                    None if previous_locator is None else str(previous_locator)
                ),
            )
            record = self._signed_record(
                record,
                initialize_new_store=previous_manifest is None,
            )
            self.operation_repository.create(record, record.canonical_bytes())
            self._emit("evidence_created", record)

            result = self.store.guarded_handler_io.publish_generation(
                staged, candidate_locator
            )
            record = self._checkpoint(record, OperationCheckpoint.CANDIDATE_PUBLISHED)
            self._emit("candidate_published", record)

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
            self.store.manifest_repository.publish_if_expected(
                key,
                expected,
                signed_manifest.canonical_bytes(),
                entry_data=self.store._manifest_entry_data(signed_manifest),
            )
            record = self._checkpoint(record, OperationCheckpoint.AUTHORITY_PUBLISHED)
            self._emit("authority_published", record)

            try:
                if previous_locator is not None and previous_locator != candidate_locator:
                    self.store._delete_or_prove_absent(previous_locator)
                record = self._checkpoint(record, OperationCheckpoint.CLEANUP_COMPLETED)
                self._emit("cleanup_completed", record)
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
