"""Deterministic S3 maintenance-evidence contracts for obstore.

Moto and boto3 provide only local bucket provisioning. The current payload
participant remains ``ObstoreGenerationIO`` and does not receive a boto3 client.
"""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest
from moto.server import ThreadedMotoServer

from cacheness.storage.obstore_generation_io import ObstoreGenerationIO


class _NativeHandler:
    """Write bytes through one handler-owned native file."""

    def put(self, value: bytes, path: Path, _config: object) -> dict[str, object]:
        artifact = path.with_suffix(".native")
        artifact.write_bytes(value)
        return {
            "actual_path": str(artifact),
            "file_size": len(value),
            "metadata": {"format": "native-test"},
        }


@pytest.fixture
def s3_maintenance_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ObstoreGenerationIO:
    """Provision one local Moto bucket without injecting boto3 into storage."""

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
    ).create_bucket(Bucket="cacheness-maintenance-contract")
    handler_root = tmp_path / "private-stage"
    handler_root.mkdir()
    provider = ObstoreGenerationIO.for_s3(
        bucket="cacheness-maintenance-contract",
        prefix="managed/run",
        region="us-east-1",
        handler_root=handler_root,
        endpoint=endpoint,
        _allow_test_endpoint=True,
        max_inventory_objects=1,
    )
    try:
        yield provider
    finally:
        provider.close()
        server.stop()


def _publish(
    provider: ObstoreGenerationIO, locator: Path, payload: bytes
) -> dict[str, object]:
    with provider.stage(_NativeHandler(), payload, object()) as staged:
        return provider.publish_generation(staged, locator)


def test_inventory_is_bounded_report_only_evidence(
    s3_maintenance_provider: ObstoreGenerationIO,
) -> None:
    """One bounded page reports only the participant's managed generations."""

    provider = s3_maintenance_provider
    first_locator = Path("generations") / "inventory-one" / "payload.native"
    second_locator = Path("generations") / "inventory-two" / "payload.native"
    _publish(provider, first_locator, b"one")
    _publish(provider, second_locator, b"two")

    first = provider.inventory_page()
    assert len(first.objects) == 1
    assert first.next_offset is not None
    second = provider.inventory_page(first.next_offset)
    assert {item.locator for item in (*first.objects, *second.objects)} == {
        first_locator.as_posix(),
        second_locator.as_posix(),
    }


def test_exact_cleanup_deletes_one_generation_or_proves_absence(
    s3_maintenance_provider: ObstoreGenerationIO,
) -> None:
    """Cleanup stays exact and idempotent without a second S3 authority."""

    provider = s3_maintenance_provider
    locator = Path("generations") / "cleanup" / "payload.native"
    _publish(provider, locator, b"owned")

    provider.delete_or_prove_absent(locator)
    provider.delete_or_prove_absent(locator)
    with pytest.raises(FileNotFoundError):
        with provider.open_snapshot(locator, {}):
            pass
