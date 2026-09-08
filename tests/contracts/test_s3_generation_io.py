"""Contract-only tests for the Amazon S3 generation-I/O participant.

Moto and local fakes exercise request construction, containment, bounds, and
typed failure classification.  They are intentionally not AWS qualification
evidence; the live-service qualification suite owns that claim.
"""

from __future__ import annotations

import stat
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
    reason="S3 generation-I/O contracts require boto3 and moto",
)


class _NativeHandler:
    """Minimal handler retaining ownership of its native payload format."""

    def put(self, value: bytes, path: Path, _config: object) -> dict[str, object]:
        artifact = path.with_suffix(".native")
        artifact.write_bytes(value)
        return {
            "actual_path": str(artifact),
            "file_size": len(value),
            "metadata": {"format": "native-test"},
        }


class _TrackedBody:
    """Delegate a streaming body while recording that the network body closed."""

    def __init__(self, body: object) -> None:
        self._body = body
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        return self._body.read(amount)

    def close(self) -> None:
        self.closed = True
        self._body.close()


class _TrackedClient:
    """Wrap Moto's client without changing its request semantics."""

    def __init__(self, client: object) -> None:
        self._client = client
        self.last_body: _TrackedBody | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self._client, name)

    def get_object(self, **kwargs: object) -> dict[str, object]:
        response = dict(self._client.get_object(**kwargs))
        body = _TrackedBody(response["Body"])
        self.last_body = body
        response["Body"] = body
        return response


class _MultipartResponseLossClient(_TrackedClient):
    """Model an S3 response loss after the service accepted multipart completion."""

    def __init__(self, client: object) -> None:
        super().__init__(client)
        self.complete_requests: list[dict[str, object]] = []

    def complete_multipart_upload(self, **kwargs: object) -> dict[str, object]:
        from botocore.exceptions import EndpointConnectionError

        self.complete_requests.append(dict(kwargs))
        self._client.complete_multipart_upload(**kwargs)
        raise EndpointConnectionError(endpoint_url="https://s3.amazonaws.com")


class _MultipartConflictOnceClient(_TrackedClient):
    """Inject one documented conditional-completion conflict before success."""

    def __init__(self, client: object) -> None:
        super().__init__(client)
        self.create_count = 0
        self.abort_requests: list[dict[str, object]] = []
        self._conflicted = False

    def create_multipart_upload(self, **kwargs: object) -> dict[str, object]:
        self.create_count += 1
        return self._client.create_multipart_upload(**kwargs)

    def abort_multipart_upload(self, **kwargs: object) -> dict[str, object]:
        self.abort_requests.append(dict(kwargs))
        return self._client.abort_multipart_upload(**kwargs)

    def complete_multipart_upload(self, **kwargs: object) -> dict[str, object]:
        from botocore.exceptions import ClientError

        if not self._conflicted:
            self._conflicted = True
            raise ClientError({"Error": {"Code": "409"}}, "CompleteMultipartUpload")
        return self._client.complete_multipart_upload(**kwargs)


class _MultipartPartFailureClient(_TrackedClient):
    """Inject an observed pre-completion part failure with an exact abort target."""

    def __init__(self, client: object) -> None:
        super().__init__(client)
        self.abort_requests: list[dict[str, object]] = []

    def upload_part(self, **kwargs: object) -> dict[str, object]:
        from botocore.exceptions import ClientError

        raise ClientError({"Error": {"Code": "InternalError"}}, "UploadPart")

    def abort_multipart_upload(self, **kwargs: object) -> dict[str, object]:
        self.abort_requests.append(dict(kwargs))
        return self._client.abort_multipart_upload(**kwargs)


@pytest.fixture
def s3_generation_backend(tmp_path: Path):
    """Build a contract-only Amazon S3 participant with an injected client."""
    from cacheness.storage.backends.s3_backend import S3BlobBackend

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="cacheness-contract-bucket")
        tracked_client = _TrackedClient(client)
        backend = S3BlobBackend(
            bucket="cacheness-contract-bucket",
            prefix="contract/run",
            client=tracked_client,
            staging_root=tmp_path / "private-stage",
        )
        try:
            yield backend, tracked_client
        finally:
            backend.close()


def test_integrity_small_generation_is_conditionally_published_and_snapshotted(
    s3_generation_backend: tuple[object, _TrackedClient],
) -> None:
    """A native file reaches a private, closed-body snapshot without overwrite."""
    from cacheness.error_handling import CacheBlobLifecycleConflictError

    backend, client = s3_generation_backend
    guarded_io = backend.materialize_handler_io()
    locator = Path("generations") / "one" / "payload.native"

    with guarded_io.stage(_NativeHandler(), b"one immutable payload", object()) as staged:
        published = guarded_io.publish_generation(staged, locator)

    assert published["file_size"] == len(b"one immutable payload")
    with guarded_io.open_snapshot(locator, dict(published["metadata"])) as snapshot:
        assert snapshot.path.read_bytes() == b"one immutable payload"
        assert stat.S_IMODE(snapshot.path.stat().st_mode) == 0o600
        assert snapshot.path.name == "snapshot.native"
        assert client.last_body is not None
        assert client.last_body.closed is True

    with guarded_io.stage(_NativeHandler(), b"replacement", object()) as staged:
        with pytest.raises(CacheBlobLifecycleConflictError):
            guarded_io.publish_generation(staged, locator)


def test_recovery_multipart_response_loss_is_verified_by_exact_digest_and_size(
    tmp_path: Path,
) -> None:
    """Response loss is success only after exact-object snapshot verification.

    This is Moto/fake contract coverage, not real-AWS qualification evidence.
    """
    from cacheness.storage.backends.s3_backend import S3BlobBackend

    payload = b"x" * (5 * 1024 * 1024 + 31)
    with mock_aws():
        moto_client = boto3.client("s3", region_name="us-east-1")
        moto_client.create_bucket(Bucket="cacheness-multipart-contract")
        client = _MultipartResponseLossClient(moto_client)
        backend = S3BlobBackend(
            bucket="cacheness-multipart-contract",
            prefix="contract/run",
            client=client,
            staging_root=tmp_path / "private-stage",
            multipart_threshold=5 * 1024 * 1024,
            part_size=5 * 1024 * 1024,
            max_upload_bytes=len(payload) + 1,
        )
        try:
            guarded_io = backend.materialize_handler_io()
            locator = Path("generations") / "multipart" / "payload.native"
            with guarded_io.stage(_NativeHandler(), payload, object()) as staged:
                published = guarded_io.publish_generation(staged, locator)

            assert published["file_size"] == len(payload)
            assert len(client.complete_requests) == 1
            assert client.complete_requests[0]["IfNoneMatch"] == "*"
            with guarded_io.open_snapshot(locator, dict(published["metadata"])) as snapshot:
                assert snapshot.path.read_bytes() == payload
        finally:
            backend.close()


def test_progress_multipart_409_restarts_one_new_bounded_upload(tmp_path: Path) -> None:
    """One 409 creates exactly one new upload rather than an unbounded retry loop."""
    from cacheness.storage.backends.s3_backend import S3BlobBackend

    payload = b"y" * (5 * 1024 * 1024 + 31)
    with mock_aws():
        moto_client = boto3.client("s3", region_name="us-east-1")
        moto_client.create_bucket(Bucket="cacheness-retry-contract")
        client = _MultipartConflictOnceClient(moto_client)
        backend = S3BlobBackend(
            bucket="cacheness-retry-contract",
            prefix="contract/run",
            client=client,
            staging_root=tmp_path / "private-stage",
            multipart_threshold=5 * 1024 * 1024,
            part_size=5 * 1024 * 1024,
            max_upload_bytes=len(payload) + 1,
            max_upload_attempts=2,
        )
        try:
            guarded_io = backend.materialize_handler_io()
            with guarded_io.stage(_NativeHandler(), payload, object()) as staged:
                guarded_io.publish_generation(
                    staged, Path("generations") / "retry" / "payload.native"
                )
            assert client.create_count == 2
            assert len(client.abort_requests) == 1
        finally:
            backend.close()


def test_recovery_part_failure_aborts_only_the_known_upload(tmp_path: Path) -> None:
    """An observed pre-completion failure has one exact abort obligation."""
    from cacheness.error_handling import CacheBlobBackendError
    from cacheness.storage.backends.s3_backend import S3BlobBackend

    payload = b"z" * (5 * 1024 * 1024 + 31)
    with mock_aws():
        moto_client = boto3.client("s3", region_name="us-east-1")
        moto_client.create_bucket(Bucket="cacheness-abort-contract")
        client = _MultipartPartFailureClient(moto_client)
        backend = S3BlobBackend(
            bucket="cacheness-abort-contract",
            prefix="contract/run",
            client=client,
            staging_root=tmp_path / "private-stage",
            multipart_threshold=5 * 1024 * 1024,
            part_size=5 * 1024 * 1024,
            max_upload_bytes=len(payload) + 1,
        )
        try:
            guarded_io = backend.materialize_handler_io()
            with guarded_io.stage(_NativeHandler(), payload, object()) as staged:
                with pytest.raises(CacheBlobBackendError, match="before completion"):
                    guarded_io.publish_generation(
                        staged, Path("generations") / "abort" / "payload.native"
                    )
            assert len(client.abort_requests) == 1
            assert client.abort_requests[0]["Key"] == "contract/run/generations/abort/payload.native"
        finally:
            backend.close()
