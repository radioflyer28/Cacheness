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
        with pytest.raises(Exception):
            guarded_io.publish_generation(staged, locator)

