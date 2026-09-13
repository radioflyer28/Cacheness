"""Pinned obstore SDK parity checks for Cacheness's consumed object mechanics.

These tests exercise the narrow synchronous SDK surface that the future payload
participant may consume.  They do not qualify real AWS, authorize object
listings as catalog truth, or add lifecycle coordination outside BlobStore.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pytest
from moto.server import ThreadedMotoServer
from obstore.exceptions import (
    AlreadyExistsError,
    NotFoundError,
    UnknownConfigurationKeyError,
)
from obstore.store import LocalStore, MemoryStore, S3Store


_REGION = "us-east-1"
_BUCKET = "cacheness-obstore-parity"
_KEY = "generations/parity/exact-payload.bin"
_PAYLOAD = b"cacheness-obstore-sdk-parity"


@dataclass(frozen=True)
class _StoreCase:
    """A named store under the exact SDK parity contract."""

    name: str
    store: object


class _RecordedStore:
    """Record the narrow object calls without changing SDK behavior."""

    def __init__(self, store: object) -> None:
        self._store = store
        self.put_calls: list[tuple[str, dict[str, object]]] = []
        self.head_calls: list[str] = []
        self.get_calls: list[str] = []
        self.list_calls: list[tuple[str | None, dict[str, object]]] = []
        self.delete_calls: list[list[str]] = []

    def put(self, path: str, source: object, **kwargs: object) -> object:
        self.put_calls.append((path, dict(kwargs)))
        return self._store.put(path, source, **kwargs)

    def head(self, path: str) -> object:
        self.head_calls.append(path)
        return self._store.head(path)

    def get(self, path: str) -> object:
        self.get_calls.append(path)
        return self._store.get(path)

    def list(self, prefix: str | None = None, **kwargs: object) -> object:
        self.list_calls.append((prefix, dict(kwargs)))
        return self._store.list(prefix, **kwargs)

    def delete(self, paths: list[str]) -> object:
        self.delete_calls.append(list(paths))
        return self._store.delete(paths)


@pytest.fixture
def moto_s3_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[_StoreCase]:
    """Serve S3 over local HTTP; boto3 provisions only the test bucket."""
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint_url = f"http://{host}:{port}"

    parsed_endpoint = urlparse(endpoint_url)
    assert parsed_endpoint.scheme == "http"
    assert parsed_endpoint.hostname == "127.0.0.1"

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=_REGION,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    client.create_bucket(Bucket=_BUCKET)

    store = S3Store(
        _BUCKET,
        prefix="sdk-parity",
        config={"region": _REGION, "conditional_put": "etag"},
        client_options={"endpoint_url": endpoint_url, "allow_http": True},
    )
    try:
        yield _StoreCase("s3", store)
    finally:
        server.stop()


@pytest.fixture(params=("local", "memory", "s3"))
def parity_store(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> _StoreCase:
    """Build each supported SDK store without sharing payload state."""
    store_name = str(request.param)
    if store_name == "local":
        return _StoreCase("local", LocalStore(tmp_path / "objects", mkdir=True))
    if store_name == "memory":
        return _StoreCase("memory", MemoryStore())
    return request.getfixturevalue("moto_s3_store")


def _typed_absence(store: _RecordedStore, key: str) -> None:
    """Assert only documented absent-object variants represent absence."""
    with pytest.raises((NotFoundError, FileNotFoundError)):
        store.head(key)


def test_consumed_sync_object_primitives_have_exact_immutable_parity(
    parity_store: _StoreCase,
) -> None:
    """Conditional create, exact recovery, streaming, bounded list, and delete work."""
    store = _RecordedStore(parity_store.store)

    result = store.put(
        _KEY,
        io.BytesIO(_PAYLOAD),
        mode="create",
        use_multipart=False,
    )
    assert hasattr(result, "e_tag")
    assert hasattr(result, "version")
    first_meta = store.head(_KEY)
    assert first_meta.size == len(_PAYLOAD)
    assert hasattr(first_meta, "e_tag")
    assert hasattr(first_meta, "version")

    # A lost create response is recovered only by this exact-object observation.
    with pytest.raises(AlreadyExistsError):
        store.put(
            _KEY,
            io.BytesIO(b"replacement bytes must never win"),
            mode="create",
            use_multipart=False,
        )
    recovered_meta = store.head(_KEY)
    assert recovered_meta.size == len(_PAYLOAD)
    assert store.list_calls == []

    stream_result = store.get(_KEY)
    assert stream_result.meta.size == len(_PAYLOAD)
    assert b"".join(stream_result.stream(min_chunk_size=1)) == _PAYLOAD

    page = list(store.list("generations/", chunk_size=1))
    assert len(page) == 1
    assert page[0].size == len(_PAYLOAD)
    assert store.list_calls == [("generations/", {"chunk_size": 1})]

    store.delete([_KEY])
    assert store.delete_calls == [[_KEY]]
    _typed_absence(store, _KEY)


def test_sdk_error_types_keep_collision_absence_and_configuration_distinct(
    moto_s3_store: _StoreCase,
) -> None:
    """D-16 is a typed opt-out, never a generic-header workaround or fallback."""
    endpoint_store = moto_s3_store.store
    assert isinstance(endpoint_store, S3Store)

    with pytest.raises(UnknownConfigurationKeyError, match="expected_bucket_owner"):
        S3Store(
            _BUCKET,
            config={
                "region": _REGION,
                "expected_bucket_owner": "123456789012",
            },
        )

    assert AlreadyExistsError is not NotFoundError
    assert not issubclass(UnknownConfigurationKeyError, NotFoundError)
