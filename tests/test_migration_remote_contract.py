"""Deterministic non-live contracts for remote migration participants.

Moto/fake coverage proves only S3 adapter mechanics.  It does not qualify a
real PostgreSQL/Amazon-S3 topology; Phase 8 owns that service evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest


try:
    import boto3
    from botocore.exceptions import BotoCoreError
    from moto import mock_aws
except ImportError:  # pragma: no cover - optional remote-test dependency
    boto3 = None
    BotoCoreError = Exception
    mock_aws = None


pytestmark = pytest.mark.skipif(
    boto3 is None or mock_aws is None,
    reason="deterministic remote migration contracts require boto3 and moto",
)


class _ResponseLostClient:
    """Commit once, then simulate a lost S3 write response without a live service."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.put_calls = 0
        self.list_calls = 0
        self._lose_first_response = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def put_object(self, **kwargs: Any) -> Any:
        self.put_calls += 1
        result = self._client.put_object(**kwargs)
        if self._lose_first_response:
            self._lose_first_response = False
            raise BotoCoreError()
        return result

    def list_objects_v2(self, **kwargs: Any) -> Any:
        self.list_calls += 1
        return self._client.list_objects_v2(**kwargs)


class _AbortFailureClient:
    """Inject one exact S3 cleanup interruption without enabling discovery."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.fail_snapshot_head = False
        self.fail_delete = False
        self.list_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def head_object(self, **kwargs: Any) -> Any:
        if self.fail_snapshot_head:
            self.fail_snapshot_head = False
            raise BotoCoreError()
        return self._client.head_object(**kwargs)

    def delete_object(self, **kwargs: Any) -> Any:
        if self.fail_delete:
            self.fail_delete = False
            raise BotoCoreError()
        return self._client.delete_object(**kwargs)

    def list_objects_v2(self, **kwargs: Any) -> Any:
        self.list_calls += 1
        return self._client.list_objects_v2(**kwargs)


class _RemoteKeyProvider:
    """Share one test signing key across deterministic source and destination stores."""

    def get_key(self) -> bytes:
        return b"r" * 32


def _remote_abort_service(
    tmp_path: Path,
    *,
    client: _AbortFailureClient,
    bucket: str,
    run_id: str,
):
    """Build non-live S3 payload mechanics with a deterministic authority fake.

    The authority preserves the existing lifecycle seam but is deliberately not
    a PostgreSQL service qualification claim.
    """
    from cacheness.storage import BackendRef, BlobStore, StoreTopology
    from cacheness.storage.backends.s3_backend import S3BlobBackend
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
    source = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
        ),
        cache_dir=tmp_path / "source",
        manifest_key_provider=key_provider,
    )
    payload = S3BlobBackend(
        bucket=bucket,
        prefix=f"managed/{run_id}",
        client=client,
        staging_root=tmp_path / "private-stage",
    )
    payload.qualification_identity = "s3"
    destination = BlobStore(
        StoreTopology(
            payload=BackendRef(instance=payload, transfer_ownership=True),
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


def test_s3_candidate_response_loss_revalidates_exact_receipt_without_listing(
    tmp_path: Path,
) -> None:
    """A deterministic/non-live response loss uses digest and size, never discovery."""
    from cacheness.storage.backends.s3_backend import (
        S3BlobBackend,
        S3MigrationCandidateReceipt,
    )

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket = "cacheness-remote-migration-contract"
        client.create_bucket(Bucket=bucket)
        observed = _ResponseLostClient(client)
        backend = S3BlobBackend(
            bucket=bucket,
            prefix="managed/remote",
            client=observed,
            staging_root=tmp_path / "private-stage",
        )
        try:
            payload = b"immutable remote candidate"
            digest = hashlib.sha256(payload).hexdigest()
            locator = backend.migration_candidate_locator(
                run_id="remote-run-1", candidate_id="candidate-1.native"
            )
            receipt = backend.materialize_handler_io().write_migration_candidate(
                run_id="remote-run-1",
                plan_digest="a" * 64,
                source_revision=7,
                locator=locator,
                payload=payload,
                payload_digest=digest,
                byte_size=len(payload),
            )

            assert receipt == S3MigrationCandidateReceipt(
                run_id="remote-run-1",
                plan_digest="a" * 64,
                source_revision=7,
                locator=locator,
                payload_digest=digest,
                byte_size=len(payload),
            )
            assert observed.put_calls == 1
            assert observed.list_calls == 0
            assert not hasattr(backend, "adopt_migration_candidate")
        finally:
            backend.close()


def test_s3_candidate_rejects_unowned_locator_before_mutating(tmp_path: Path) -> None:
    """A matching object outside the exact run marker cannot be adopted or purged."""
    from cacheness.error_handling import CacheBlobLifecycleConflictError
    from cacheness.storage.backends.s3_backend import S3BlobBackend

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket = "cacheness-remote-migration-ownership"
        client.create_bucket(Bucket=bucket)
        observed = _ResponseLostClient(client)
        backend = S3BlobBackend(
            bucket=bucket,
            prefix="managed/remote",
            client=observed,
            staging_root=tmp_path / "private-stage",
        )
        try:
            with pytest.raises(CacheBlobLifecycleConflictError, match="ownership"):
                backend.materialize_handler_io().write_migration_candidate(
                    run_id="remote-run-1",
                    plan_digest="a" * 64,
                    source_revision=7,
                    locator="migration-candidates/other-run/candidate.native",
                    payload=b"immutable remote candidate",
                    payload_digest=hashlib.sha256(b"immutable remote candidate").hexdigest(),
                    byte_size=len(b"immutable remote candidate"),
                )
            assert observed.put_calls == 0
            assert observed.list_calls == 0
        finally:
            backend.close()


def test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry(
    tmp_path: Path,
) -> None:
    """S3 cleanup failures remain attributed debt until explicit retry settles them.

    Moto verifies payload-participant behavior only; the in-memory authority
    fake is not real PostgreSQL/Amazon-S3 qualification evidence.
    """
    from cacheness.storage.migration_evidence import MaintenanceEvidenceState

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket = "cacheness-remote-migration-abort"
        client.create_bucket(Bucket=bucket)
        observed = _AbortFailureClient(client)

        source, destination, service, plan = _remote_abort_service(
            tmp_path / "snapshot",
            client=observed,
            bucket=bucket,
            run_id="snapshot-failure",
        )
        try:
            candidates = destination.lifecycle_authority.candidate_entries_for_run(
                run_id=service.run_id
            )
            assert len(candidates) == 1
            observed.fail_snapshot_head = True

            partial = service.abort(plan)

            assert partial.state is MaintenanceEvidenceState.STAGING
            assert partial.deleted_entries == 0
            assert len(service.read_evidence().cleanup_debt) == 1
            assert observed.list_calls == 0

            completed = service.abort(plan)
            assert completed.state is MaintenanceEvidenceState.ABORTED
            assert completed.deleted_entries == 1
            assert service.read_evidence().cleanup_debt == ()
            assert observed.list_calls == 0
        finally:
            source.close()
            destination.close()

        source, destination, service, plan = _remote_abort_service(
            tmp_path / "delete",
            client=observed,
            bucket=bucket,
            run_id="delete-failure",
        )
        try:
            observed.fail_delete = True

            partial = service.abort(plan)

            assert partial.state is MaintenanceEvidenceState.STAGING
            assert partial.deleted_entries == 0
            assert len(service.read_evidence().cleanup_debt) == 1
            assert observed.list_calls == 0

            completed = service.abort(plan)
            assert completed.state is MaintenanceEvidenceState.ABORTED
            assert completed.deleted_entries == 1
            assert service.read_evidence().cleanup_debt == ()
            assert observed.list_calls == 0
        finally:
            source.close()
            destination.close()
