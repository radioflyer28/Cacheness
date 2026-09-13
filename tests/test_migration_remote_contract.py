"""Deterministic non-live contracts for remote migration participants.

Moto/fake coverage proves only S3 adapter mechanics.  It does not qualify a
real PostgreSQL/Amazon-S3 topology; Phase 8 owns that service evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from obstore.store import MemoryStore

from cacheness.error_handling import CacheBlobBackendError, CacheUnsafePathError
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.obstore_generation_io import ObstoreGenerationIO


class _AbortFailureParticipant(ObstoreGenerationIO):
    """Inject exact remote-style cleanup interruptions without discovery access."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True)
        super().__init__(
            MemoryStore(), GuardedHandlerIO(root), qualification_identity="s3"
        )
        self.fail_snapshot = False
        self.fail_delete = False
        self.inventory_calls = 0

    def open_snapshot(self, locator, metadata):
        if self.fail_snapshot:
            self.fail_snapshot = False
            raise CacheBlobBackendError("injected exact remote snapshot failure")
        return super().open_snapshot(locator, metadata)

    def delete_or_prove_absent(self, locator):
        if self.fail_delete:
            self.fail_delete = False
            raise CacheBlobBackendError("injected exact remote deletion failure")
        return super().delete_or_prove_absent(locator)

    def inventory_page(self, *args, **kwargs):
        self.inventory_calls += 1
        return super().inventory_page(*args, **kwargs)


class _RemoteKeyProvider:
    """Share one test signing key across deterministic source and destination stores."""

    def get_key(self) -> bytes:
        return b"r" * 32


class _ParticipantPayloadHandler:
    """Minimal path-based payload handler for participant-only contracts."""

    def put(self, value: object, file_path: Path, config: object) -> dict[str, object]:
        del config
        assert isinstance(value, dict)
        payload = value["value"].encode("utf-8")
        target = file_path.with_suffix(".remote")
        target.write_bytes(payload)
        return {"actual_path": str(target), "file_size": len(payload)}


def _remote_abort_service(
    tmp_path: Path,
    *,
    participant: _AbortFailureParticipant,
    run_id: str,
):
    """Build non-live S3 payload mechanics with a deterministic authority fake.

    The authority preserves the existing lifecycle seam but is deliberately not
    a PostgreSQL service qualification claim.
    """
    from cacheness.storage import BackendRef, BlobStore, StoreTopology
    from cacheness.storage.guarded_handler_io import GuardedHandlerIO
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
    from cacheness.storage.migration import (
        MigrationCompatibilityEdge,
        OfflineMigrationService,
    )
    from cacheness.storage.manifest import StoreVersionDimensions

    class _DeterministicPostgresqlAuthority(InMemoryLifecycleAuthority):
        qualification_identity = "postgresql"

    class _RemotePayloadHandler:
        """One native test format shared by both sides of the migration edge."""

        @property
        def data_type(self) -> str:
            return "remote-migration-payload"

        @property
        def payload_format(self) -> str:
            return "remote-migration-payload"

        @property
        def payload_format_version(self) -> int:
            return 1

        def supports_payload_contract(self, payload_format: str, version: int) -> bool:
            return (payload_format, version) == (
                self.payload_format,
                self.payload_format_version,
            )

        def payload_transformation_edges(self) -> tuple[object, ...]:
            return ()

        def can_handle(self, value: object, config: object = None) -> bool:
            del config
            return isinstance(value, dict) and isinstance(value.get("value"), str)

        def put(self, value: object, file_path: Path, config: object) -> dict[str, object]:
            del config
            assert isinstance(value, dict)
            payload = value["value"].encode("utf-8")
            target = file_path.with_suffix(".remote")
            target.write_bytes(payload)
            return {
                "actual_path": str(target),
                "file_size": len(payload),
                "payload_format": self.payload_format,
                "payload_format_version": self.payload_format_version,
                "metadata": {"native": "remote"},
            }

        def get(self, file_path: Path, metadata: object) -> dict[str, str]:
            del metadata
            return {"value": file_path.read_text(encoding="utf-8")}

        def get_file_extension(self, config: object) -> str:
            del config
            return ".remote"

    key_provider = _RemoteKeyProvider()
    (tmp_path / "source-private-stage").mkdir(parents=True)
    source_payload = ObstoreGenerationIO(
        MemoryStore(),
        GuardedHandlerIO(tmp_path / "source-private-stage"),
        qualification_identity="s3",
    )
    source = BlobStore(
        StoreTopology(
            payload=BackendRef(instance=source_payload, transfer_ownership=True),
            authority=BackendRef(
                instance=_DeterministicPostgresqlAuthority(), transfer_ownership=True
            ),
        ),
        cache_dir=tmp_path / "source",
        manifest_key_provider=key_provider,
    )
    destination = BlobStore(
        StoreTopology(
            payload=BackendRef(instance=participant, transfer_ownership=True),
            authority=BackendRef(
                instance=_DeterministicPostgresqlAuthority(), transfer_ownership=True
            ),
        ),
        cache_dir=tmp_path / "destination",
        manifest_key_provider=key_provider,
    )
    source.initialize()
    destination.initialize()
    source.handlers.register_handler(_RemotePayloadHandler(), priority=0)
    destination.handlers.register_handler(_RemotePayloadHandler(), priority=0)
    source.put_entry({"value": run_id}, key="entry")
    service = OfflineMigrationService(
        source=source,
        destination=destination,
        work_directory=tmp_path / "maintenance",
        run_id=run_id,
        stopped_workers_acknowledged=True,
        compatibility_edges=(
            MigrationCompatibilityEdge(
                source=StoreVersionDimensions(payload_format_version=1),
                destination=StoreVersionDimensions(payload_format_version=1),
                name="current-object-contract",
            ),
        ),
    )
    plan = service.plan(service.inspect())
    service.stage(plan)
    return source, destination, service, plan


def test_remote_participant_publishes_one_exact_generation_without_inventory(
    tmp_path: Path,
) -> None:
    """The remote-shaped participant owns immutable effects, never discovery."""
    participant = _AbortFailureParticipant(tmp_path / "private-stage")
    handler = _ParticipantPayloadHandler()
    try:
        with participant.stage(handler, {"value": "immutable remote candidate"}, None) as staged:
            receipt = participant.publish_generation(
                staged, "generations/remote-run/candidate-1.remote"
            )
        assert receipt["actual_path"] == "generations/remote-run/candidate-1.remote"
        assert participant.inventory_calls == 0
    finally:
        participant.close()


def test_remote_participant_rejects_outside_generation_namespace_before_mutating(
    tmp_path: Path,
) -> None:
    """An exact participant rejects a path outside its immutable namespace."""
    participant = _AbortFailureParticipant(tmp_path / "private-stage")
    handler = _ParticipantPayloadHandler()
    try:
        with participant.stage(handler, {"value": "immutable remote candidate"}, None) as staged:
            with pytest.raises(CacheUnsafePathError, match="immutable namespace"):
                participant.publish_generation(
                    staged, "migration-candidates/other-run/candidate.remote"
                )
        assert participant.inventory_calls == 0
    finally:
        participant.close()


def test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry(
    tmp_path: Path,
) -> None:
    """S3 cleanup failures remain attributed debt until explicit retry settles them.

    The in-memory object adapter exercises only participant mechanics; the
    in-memory authority fake is not PostgreSQL/Amazon-S3 qualification proof.
    """
    from cacheness.storage.migration_evidence import MaintenanceEvidenceState

    participant = _AbortFailureParticipant(tmp_path / "snapshot-private-stage")
    source, destination, service, plan = _remote_abort_service(
        tmp_path / "snapshot", participant=participant, run_id="snapshot-failure"
    )
    try:
        candidates = destination.lifecycle_authority.candidate_entries_for_run(
            run_id=service.run_id
        )
        assert len(candidates) == 1
        participant.fail_snapshot = True

        partial = service.abort(plan)

        assert partial.state is MaintenanceEvidenceState.STAGING
        assert partial.deleted_entries == 0
        assert len(service.read_evidence().cleanup_debt) == 1
        assert participant.inventory_calls == 0

        completed = service.abort(plan)
        assert completed.state is MaintenanceEvidenceState.ABORTED
        assert completed.deleted_entries == 1
        assert service.read_evidence().cleanup_debt == ()
        assert participant.inventory_calls == 0
    finally:
        source.close()
        destination.close()

    participant = _AbortFailureParticipant(tmp_path / "delete-private-stage")
    source, destination, service, plan = _remote_abort_service(
        tmp_path / "delete", participant=participant, run_id="delete-failure"
    )
    try:
        participant.fail_delete = True

        partial = service.abort(plan)

        assert partial.state is MaintenanceEvidenceState.STAGING
        assert partial.deleted_entries == 0
        assert len(service.read_evidence().cleanup_debt) == 1
        assert participant.inventory_calls == 0

        completed = service.abort(plan)
        assert completed.state is MaintenanceEvidenceState.ABORTED
        assert completed.deleted_entries == 1
        assert service.read_evidence().cleanup_debt == ()
        assert participant.inventory_calls == 0
    finally:
        source.close()
        destination.close()
