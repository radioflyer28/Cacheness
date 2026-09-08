"""Real Amazon S3 coverage for immutable Phase 5 generation primitives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheUnsafePathError,
)
from cacheness.storage.backends.s3_backend import S3BlobBackend


pytestmark = pytest.mark.live_aws_s3


class _NativeHandler:
    """Write bytes in one handler-owned native payload format."""

    def put(self, value: bytes, path: Path, _config: object) -> dict[str, object]:
        artifact = path.with_suffix(".native")
        artifact.write_bytes(value)
        return {
            "actual_path": str(artifact),
            "file_size": len(value),
            "metadata": {"storage_format": "live-native"},
        }


def _backend(
    resources: Any,
    client: Any,
    tmp_path: Path,
    *,
    prefix: str | None = None,
    multipart_threshold: int = 8 * 1024 * 1024,
    max_inventory_objects: int = 1000,
) -> S3BlobBackend:
    """Construct a real-service participant scoped below the owned run prefix."""
    return S3BlobBackend(
        bucket=resources.config.s3_bucket,
        prefix=prefix or resources.namespace.prefix,
        region=resources.config.region or "us-east-1",
        client=client,
        expected_bucket_owner=resources.config.expected_bucket_owner,
        staging_root=tmp_path / "private-stage",
        multipart_threshold=multipart_threshold,
        part_size=5 * 1024 * 1024,
        max_upload_bytes=8 * 1024 * 1024,
        max_download_bytes=8 * 1024 * 1024,
        max_download_work=256,
        max_inventory_objects=max_inventory_objects,
        max_inventory_bytes=8 * 1024 * 1024,
        max_inventory_work=1,
    )


def test_live_s3_small_generation_is_conditional_and_exactly_verified(
    live_qualification_resources: Any,
    live_s3_client: Any,
    tmp_path: Path,
) -> None:
    """A real small object is immutable, snapshot-contained, and digest-verified."""
    resources = live_qualification_resources
    backend = _backend(resources, live_s3_client, tmp_path)
    guarded_io = backend.materialize_handler_io()
    locator = Path("small") / uuid4().hex / "payload.native"
    payload = b"real-s3-small-generation"
    try:
        with guarded_io.stage(_NativeHandler(), payload, object()) as staged:
            published = guarded_io.publish_generation(staged, locator)
        with guarded_io.open_snapshot(locator, dict(published["metadata"])) as snapshot:
            assert snapshot.path.read_bytes() == payload
            assert hashlib.sha256(snapshot.path.read_bytes()).hexdigest() == hashlib.sha256(
                payload
            ).hexdigest()
            assert snapshot.path.stat().st_size == len(payload)

        with guarded_io.stage(_NativeHandler(), b"replacement", object()) as staged:
            with pytest.raises(CacheBlobLifecycleConflictError):
                guarded_io.publish_generation(staged, locator)
        guarded_io.delete_or_prove_absent(locator)
        with pytest.raises(FileNotFoundError):
            with guarded_io.open_snapshot(locator, {}):
                pass
        with pytest.raises(CacheUnsafePathError):
            guarded_io.open_snapshot(Path("..") / "outside.native", {})
    finally:
        backend.close()


def test_live_s3_multipart_generation_is_conditional_and_ambiguous_response_check_is_exact(
    live_qualification_resources: Any,
    live_s3_client: Any,
    tmp_path: Path,
) -> None:
    """A real multipart object verifies one exact key without using listing as truth."""
    resources = live_qualification_resources
    backend = _backend(
        resources,
        live_s3_client,
        tmp_path,
        multipart_threshold=5 * 1024 * 1024,
    )
    guarded_io = backend.materialize_handler_io()
    locator = Path("multipart") / uuid4().hex / "payload.native"
    payload = b"m" * (5 * 1024 * 1024 + 1024)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        with guarded_io.stage(_NativeHandler(), payload, object()) as staged:
            published = guarded_io.publish_generation(staged, locator)
        assert published["file_size"] == len(payload)

        # Network-response loss cannot be safely induced against the service.
        # Verify the adapter's exact-key observation against the accepted object.
        guarded_io._resolve_ambiguous_publication(
            locator.as_posix(), digest, len(payload), ConnectionError("response lost")
        )
        with pytest.raises(CacheBlobLifecycleConflictError):
            guarded_io._resolve_ambiguous_publication(
                locator.as_posix(), "0" * 64, len(payload), ConnectionError("response lost")
            )
        with guarded_io.stage(_NativeHandler(), payload, object()) as staged:
            with pytest.raises(CacheBlobLifecycleConflictError):
                guarded_io.publish_generation(staged, locator)
        guarded_io.delete_or_prove_absent(locator)
    finally:
        backend.close()


def test_live_s3_inventory_is_page_bounded_and_exact_prefix_contained(
    live_qualification_resources: Any,
    live_s3_client: Any,
    tmp_path: Path,
) -> None:
    """Continuation pages enumerate only an exact run-owned subprefix."""
    resources = live_qualification_resources
    prefix = f"{resources.namespace.prefix}inventory/{uuid4().hex}/"
    backend = _backend(
        resources, live_s3_client, tmp_path, prefix=prefix, max_inventory_objects=1
    )
    guarded_io = backend.materialize_handler_io()
    locators = (Path("one.native"), Path("two.native"))
    try:
        for locator in locators:
            with guarded_io.stage(_NativeHandler(), locator.name.encode("utf-8"), object()) as staged:
                guarded_io.publish_generation(staged, locator)
        first = guarded_io.inventory_page()
        assert len(first.objects) == 1
        assert first.next_token is not None
        second = guarded_io.inventory_page(first.next_token)
        observed = {item.locator for item in (*first.objects, *second.objects)}
        assert observed == {locator.as_posix() for locator in locators}
        assert all(not item.locator.startswith("../") for item in (*first.objects, *second.objects))
        for locator in locators:
            guarded_io.delete_or_prove_absent(locator)
    finally:
        backend.close()


def test_live_s3_rejects_untrusted_credential_request_against_the_real_service(
    live_qualification_resources: Any,
) -> None:
    """A separately signed request without the fixture identity cannot read the bucket."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    resources = live_qualification_resources
    untrusted_client = boto3.client(
        "s3",
        region_name=resources.config.region or "us-east-1",
        aws_access_key_id="invalid-phase5-access-key",
        aws_secret_access_key="invalid-phase5-secret-key",
        config=Config(connect_timeout=5, read_timeout=5, retries={"max_attempts": 0}),
    )
    with pytest.raises(ClientError):
        untrusted_client.head_bucket(Bucket=resources.config.s3_bucket)
