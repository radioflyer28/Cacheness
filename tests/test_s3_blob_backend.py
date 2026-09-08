"""Contract-only tests for S3 bounded evidence and exact cleanup primitives.

These Moto/fake tests verify adapter mechanics only. They neither exercise an
authority lifecycle transition nor qualify Amazon S3 for the remote topology.
"""

from __future__ import annotations

from pathlib import Path

import pytest


try:
    import boto3
    from moto import mock_aws
except ImportError:  # pragma: no cover - optional dependency boundary
    boto3 = None
    mock_aws = None


pytestmark = pytest.mark.skipif(
    boto3 is None or mock_aws is None,
    reason="S3 participant contracts require boto3 and moto",
)


class _ObservedClient:
    """Capture exact cleanup requests without changing Moto's behavior."""

    def __init__(self, client: object) -> None:
        self._client = client
        self.delete_requests: list[dict[str, object]] = []
        self.head_requests: list[dict[str, object]] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self._client, name)

    def delete_object(self, **kwargs: object) -> dict[str, object]:
        self.delete_requests.append(dict(kwargs))
        return self._client.delete_object(**kwargs)

    def head_object(self, **kwargs: object) -> dict[str, object]:
        self.head_requests.append(dict(kwargs))
        return self._client.head_object(**kwargs)


@pytest.fixture
def s3_participant(tmp_path: Path):
    """Build one isolated managed-prefix participant for Moto contract checks."""
    from cacheness.storage.backends.s3_backend import S3BlobBackend

    with mock_aws():
        moto_client = boto3.client("s3", region_name="us-east-1")
        bucket = "cacheness-evidence-contract"
        moto_client.create_bucket(Bucket=bucket)
        client = _ObservedClient(moto_client)
        backend = S3BlobBackend(
            bucket=bucket,
            prefix="managed/run",
            client=client,
            staging_root=tmp_path / "private-stage",
            max_inventory_objects=2,
            max_inventory_bytes=32,
            max_inventory_work=1,
        )
        try:
            yield backend, client, bucket
        finally:
            backend.close()


def test_inventory_returns_only_one_bounded_continuation_page(s3_participant) -> None:
    """Inventory never accumulates the bucket and validates the managed prefix."""
    backend, client, bucket = s3_participant
    guarded_io = backend.materialize_handler_io()
    client.put_object(Bucket=bucket, Key="managed/run/generations/a", Body=b"a")
    client.put_object(Bucket=bucket, Key="managed/run/generations/b", Body=b"bb")
    client.put_object(Bucket=bucket, Key="managed/run/generations/c", Body=b"ccc")
    client.put_object(Bucket=bucket, Key="outside/never-list", Body=b"not-owned")

    first = guarded_io.inventory_page()
    assert tuple(item.locator for item in first.objects) == (
        "generations/a",
        "generations/b",
    )
    assert first.next_token is not None

    second = guarded_io.inventory_page(first.next_token)
    assert tuple(item.locator for item in second.objects) == ("generations/c",)
    assert second.next_token is None


def test_recovery_delete_proves_exact_absence_with_one_followup_head(
    s3_participant,
) -> None:
    """Delete success is never inferred from an S3 acknowledgement alone."""
    backend, client, bucket = s3_participant
    guarded_io = backend.materialize_handler_io()
    locator = Path("generations") / "cleanup" / "one"
    key = "managed/run/generations/cleanup/one"
    client.put_object(Bucket=bucket, Key=key, Body=b"owned")

    guarded_io.delete_or_prove_absent(locator)
    assert client.delete_requests == [{"Bucket": bucket, "Key": key}]
    assert client.head_requests == [{"Bucket": bucket, "Key": key}]

    guarded_io.delete_or_prove_absent(locator)
    assert len(client.delete_requests) == 2
    assert len(client.head_requests) == 2


def test_configuration_rejects_legacy_endpoint_and_unmanaged_prefix(tmp_path: Path) -> None:
    """Pre-production cutover keeps only Amazon-S3 managed-prefix construction."""
    from cacheness.error_handling import CacheConfigurationError
    from cacheness.storage.backends.s3_backend import S3BlobBackend

    with pytest.raises(CacheConfigurationError):
        S3BlobBackend(bucket="bucket", prefix="")
    with pytest.raises(TypeError):
        S3BlobBackend(
            bucket="bucket",
            prefix="managed",
            endpoint_url="http://legacy-compatible-endpoint",
        )
    assert not hasattr(S3BlobBackend, "write_blob")
    assert not hasattr(S3BlobBackend, "read_blob")


def test_inventory_service_error_is_typed_not_an_empty_result(s3_participant, monkeypatch) -> None:
    """A remote list error does not collapse into false evidence of absence."""
    from botocore.exceptions import ClientError
    from cacheness.error_handling import CacheBlobBackendError

    backend, client, _bucket = s3_participant
    guarded_io = backend.materialize_handler_io()

    def fail_list(**_kwargs: object) -> dict[str, object]:
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2")

    monkeypatch.setattr(client, "list_objects_v2", fail_list)
    with pytest.raises(CacheBlobBackendError, match="inventory"):
        guarded_io.inventory_page()
