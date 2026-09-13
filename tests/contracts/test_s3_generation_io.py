"""Moto contracts for the current obstore S3 generation participant.

The boto3 client exists only to provision the local Moto bucket. The
participant itself uses obstore and its normal object-store construction path.
These tests are deterministic adapter coverage, not live-AWS qualification.
"""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest
from moto.server import ThreadedMotoServer

from cacheness.error_handling import CacheConfigurationError
from cacheness.storage.obstore_generation_io import ObstoreGenerationIO


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


@pytest.fixture
def s3_generation_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ObstoreGenerationIO:
    """Provision one isolated Moto bucket for the obstore S3 participant."""

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    ).create_bucket(Bucket="cacheness-generation-contract")
    handler_root = tmp_path / "private-stage"
    handler_root.mkdir()
    provider = ObstoreGenerationIO.for_s3(
        bucket="cacheness-generation-contract",
        prefix="contract/run",
        region="us-east-1",
        handler_root=handler_root,
        endpoint=endpoint,
        _allow_test_endpoint=True,
    )
    try:
        yield provider
    finally:
        provider.close()
        server.stop()


def test_integrity_s3_generation_is_conditionally_published_and_snapshotted(
    s3_generation_provider: ObstoreGenerationIO,
) -> None:
    """One native payload is immutable and materializes only a private snapshot."""

    locator = Path("generations") / "one" / "payload.native"
    provider = s3_generation_provider

    with provider.stage(_NativeHandler(), b"one immutable payload", object()) as staged:
        published = provider.publish_generation(staged, locator)

    assert published["file_size"] == len(b"one immutable payload")
    with provider.open_snapshot(locator, dict(published["metadata"])) as snapshot:
        assert snapshot.path.read_bytes() == b"one immutable payload"
        assert snapshot.path.name == "snapshot.native"

    with provider.stage(_NativeHandler(), b"replacement", object()) as staged:
        with pytest.raises(FileExistsError):
            provider.publish_generation(staged, locator)


def test_s3_generation_preserves_opaque_transport_observation(
    s3_generation_provider: ObstoreGenerationIO,
) -> None:
    """Exact S3 head evidence is available without becoming lifecycle truth."""

    locator = Path("generations") / "evidence" / "payload.native"
    provider = s3_generation_provider
    with provider.stage(_NativeHandler(), b"opaque evidence", object()) as staged:
        published = provider.publish_generation(staged, locator)

    observation = provider.observe_transport(locator.as_posix())
    assert observation.byte_size == published["file_size"]
    assert observation.e_tag is not None


def test_s3_factory_rejects_legacy_endpoint_and_owner_pinning(tmp_path: Path) -> None:
    """Only the explicit loopback Moto endpoint is allowed in deterministic tests."""

    with pytest.raises(CacheConfigurationError):
        ObstoreGenerationIO.for_s3(
            bucket="bucket",
            prefix="managed",
            region="us-east-1",
            handler_root=tmp_path,
            endpoint="https://legacy-compatible-endpoint",
            _allow_test_endpoint=True,
        )
    with pytest.raises(CacheConfigurationError, match="expected_bucket_owner"):
        ObstoreGenerationIO.for_s3(
            bucket="bucket",
            prefix="managed",
            region="us-east-1",
            handler_root=tmp_path,
            expected_bucket_owner="123456789012",
        )
