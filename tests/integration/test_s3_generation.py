"""Real Amazon S3 coverage for immutable Phase 5 generation primitives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from cacheness.error_handling import CacheUnsafePathError
from cacheness.storage.obstore_generation_io import ObstoreGenerationIO


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


def _provider(
    resources: Any,
    tmp_path: Path,
    *,
    prefix: str | None = None,
    max_inventory_objects: int = 1000,
) -> ObstoreGenerationIO:
    """Construct a real-service participant scoped below the owned run prefix."""
    handler_root = tmp_path / "private-stage"
    handler_root.mkdir(parents=True, exist_ok=True)
    return ObstoreGenerationIO.for_s3(
        bucket=resources.config.s3_bucket,
        prefix=prefix or resources.namespace.prefix,
        region=resources.config.region or "us-east-1",
        handler_root=handler_root,
        max_upload_bytes=8 * 1024 * 1024,
        max_download_bytes=8 * 1024 * 1024,
        max_inventory_objects=max_inventory_objects,
        max_inventory_bytes=8 * 1024 * 1024,
    )


def test_live_s3_small_generation_is_conditional_and_exactly_verified(
    live_qualification_resources: Any,
    tmp_path: Path,
) -> None:
    """A real small object is immutable, snapshot-contained, and digest-verified."""
    resources = live_qualification_resources
    provider = _provider(resources, tmp_path)
    locator = Path("generations") / uuid4().hex / "payload.native"
    payload = b"real-s3-small-generation"
    try:
        with provider.stage(_NativeHandler(), payload, object()) as staged:
            published = provider.publish_generation(staged, locator)
        with provider.open_snapshot(locator, dict(published["metadata"])) as snapshot:
            assert snapshot.path.read_bytes() == payload
            assert hashlib.sha256(snapshot.path.read_bytes()).hexdigest() == hashlib.sha256(
                payload
            ).hexdigest()
            assert snapshot.path.stat().st_size == len(payload)

        with provider.stage(_NativeHandler(), b"replacement", object()) as staged:
            with pytest.raises(FileExistsError):
                provider.publish_generation(staged, locator)
        provider.delete_or_prove_absent(locator)
        with pytest.raises(FileNotFoundError):
            with provider.open_snapshot(locator, {}):
                pass
        with pytest.raises(CacheUnsafePathError):
            provider.open_snapshot(Path("..") / "outside.native", {})
    finally:
        provider.close()


def test_live_s3_generation_is_conditional_and_exact_head_evidence_is_opaque(
    live_qualification_resources: Any,
    tmp_path: Path,
) -> None:
    """A real object exposes only opaque exact transport evidence."""
    resources = live_qualification_resources
    provider = _provider(resources, tmp_path)
    locator = Path("generations") / uuid4().hex / "payload.native"
    payload = b"real-s3-exact-observation"
    try:
        with provider.stage(_NativeHandler(), payload, object()) as staged:
            published = provider.publish_generation(staged, locator)
        assert published["file_size"] == len(payload)
        observation = provider.observe_transport(locator.as_posix())
        assert observation.byte_size == len(payload)
        assert observation.e_tag is not None
        with provider.stage(_NativeHandler(), payload, object()) as staged:
            with pytest.raises(FileExistsError):
                provider.publish_generation(staged, locator)
        provider.delete_or_prove_absent(locator)
    finally:
        provider.close()


def test_live_s3_inventory_is_page_bounded_and_exact_prefix_contained(
    live_qualification_resources: Any,
    tmp_path: Path,
) -> None:
    """Continuation pages enumerate only an exact run-owned subprefix."""
    resources = live_qualification_resources
    prefix = f"{resources.namespace.prefix.rstrip('/')}/inventory/{uuid4().hex}"
    provider = _provider(
        resources, tmp_path, prefix=prefix, max_inventory_objects=1
    )
    locators = (
        Path("generations") / "inventory-one" / "payload.native",
        Path("generations") / "inventory-two" / "payload.native",
    )
    try:
        for locator in locators:
            with provider.stage(_NativeHandler(), locator.name.encode("utf-8"), object()) as staged:
                provider.publish_generation(staged, locator)
        first = provider.inventory_page()
        assert len(first.objects) == 1
        assert first.next_offset is not None
        second = provider.inventory_page(first.next_offset)
        observed = {item.locator for item in (*first.objects, *second.objects)}
        assert observed == {locator.as_posix() for locator in locators}
        assert all(not item.locator.startswith("../") for item in (*first.objects, *second.objects))
        for locator in locators:
            provider.delete_or_prove_absent(locator)
    finally:
        provider.close()
