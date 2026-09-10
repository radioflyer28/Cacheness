"""Prove fail-closed locators, safe suffixes, and exact generation deletion."""

from __future__ import annotations

import inspect
import json
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import boto3
from moto.server import ThreadedMotoServer
from obstore.exceptions import NotFoundError
from obstore.store import LocalStore, MemoryStore, S3Store


SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SAFE_SUFFIX = re.compile(r"(?:\.[A-Za-z0-9_-]+){0,4}\Z")


def validate_locator(locator: str) -> str:
    if not isinstance(locator, str) or not locator or len(locator) > 512:
        raise ValueError("invalid managed locator")
    if locator.startswith("/") or "\\" in locator or ":" in locator:
        raise ValueError("invalid managed locator")
    parts = locator.split("/")
    if parts[0] != "generations" or any(
        not SAFE_SEGMENT.fullmatch(part) for part in parts
    ):
        raise ValueError("invalid managed locator")
    normalized = PurePosixPath(locator)
    if normalized.is_absolute() or normalized.parts != tuple(parts):
        raise ValueError("invalid managed locator")
    return normalized.as_posix()


def validate_suffix(suffix: str) -> str:
    if not isinstance(suffix, str) or len(suffix) > 96:
        raise ValueError("invalid native suffix")
    if not SAFE_SUFFIX.fullmatch(suffix):
        raise ValueError("invalid native suffix")
    return suffix


def is_absent(store: Any, locator: str) -> bool:
    try:
        store.head(locator)
    except (FileNotFoundError, NotFoundError):
        return True
    return False


class ManagedParticipant:
    """Narrow spike wrapper; raw obstore paths never cross this boundary."""

    def __init__(self, store: Any):
        self.store = store

    def create(self, locator: str, payload: bytes) -> None:
        self.store.put(validate_locator(locator), payload, mode="create")

    def read(self, locator: str) -> bytes:
        return bytes(self.store.get(validate_locator(locator)).bytes())

    def delete_or_prove_absent(self, locator: str) -> str:
        safe = validate_locator(locator)
        try:
            self.store.delete(safe)
        except (FileNotFoundError, NotFoundError):
            pass
        if is_absent(self.store, safe):
            return "ABSENT"
        raise OSError("exact generation remains after delete")


def exercise(name: str, store: Any) -> dict[str, Any]:
    participant = ManagedParticipant(store)
    first = "generations/key-generation-001.bin"
    second = "generations/key-generation-002.bin"
    participant.create(first, b"generation one")
    participant.create(second, b"generation two")

    invalid_locators = (
        "",
        "/absolute",
        "../escape",
        "generations/../escape",
        "generations//escape",
        "generations\\escape",
        "s3://other-bucket/object",
        "other-prefix/object",
        "generations/bad space",
    )
    rejected = 0
    for candidate in invalid_locators:
        try:
            participant.delete_or_prove_absent(candidate)
        except ValueError:
            rejected += 1
        else:  # pragma: no cover - every attack must fail closed
            raise AssertionError(f"unsafe locator accepted: {candidate!r}")
    assert participant.read(first) == b"generation one"
    assert participant.read(second) == b"generation two"

    assert participant.delete_or_prove_absent(first) == "ABSENT"
    assert is_absent(store, first)
    assert participant.read(second) == b"generation two"
    assert participant.delete_or_prove_absent(first) == "ABSENT"

    delete_parameters = tuple(inspect.signature(store.delete).parameters)
    assert delete_parameters == ("paths",)
    return {
        "store": name,
        "invalid_locators_rejected": rejected,
        "exact_generation_deleted": True,
        "sibling_generation_preserved": True,
        "repeat_delete_normalized_to_absent": True,
        "conditional_delete_parameters": [],
    }


@contextmanager
def mocked_s3() -> Iterator[S3Store]:
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    client.create_bucket(Bucket="cacheness-containment-spike")
    try:
        yield S3Store(
            "cacheness-containment-spike",
            prefix="managed",
            region="us-east-1",
            endpoint=endpoint,
            access_key_id="test",
            secret_access_key="test",
            virtual_hosted_style_request=False,
            client_options={"allow_http": True},
        )
    finally:
        server.stop()


def main() -> None:
    valid_suffixes = ("", ".npz", ".pkl.zstd", ".mcap", ".a_b-2")
    invalid_suffixes = (
        ".",
        "/absolute",
        ".tar/../../escape",
        ".bad space",
        ".é",
        "." + "a" * 97,
        ".a.b.c.d.e",
    )
    assert all(validate_suffix(item) == item for item in valid_suffixes)
    suffix_rejections = 0
    for candidate in invalid_suffixes:
        try:
            validate_suffix(candidate)
        except ValueError:
            suffix_rejections += 1
        else:  # pragma: no cover
            raise AssertionError(f"unsafe suffix accepted: {candidate!r}")

    with tempfile.TemporaryDirectory(prefix="obstore-delete-spike-") as root:
        outside = Path(root) / "outside-sentinel"
        outside.write_bytes(b"must survive")
        with mocked_s3() as s3:
            results = [
                exercise("memory", MemoryStore()),
                exercise("local", LocalStore(Path(root) / "objects", mkdir=True)),
                exercise("mocked-s3", s3),
            ]
        assert outside.read_bytes() == b"must survive"

    print(
        json.dumps(
            {
                "verdict": "PARTIAL",
                "results": results,
                "suffixes": {
                    "valid": len(valid_suffixes),
                    "invalid_rejected": suffix_rejections,
                },
                "limitation": (
                    "obstore delete is path-only and has no version/e-tag precondition; "
                    "safety depends on exact immutable locators never being reused"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

