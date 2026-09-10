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
