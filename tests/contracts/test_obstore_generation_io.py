"""Public contracts for obstore-backed payload generation participants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pytest
from moto.server import ThreadedMotoServer

from obstore.exceptions import PreconditionError
from obstore.store import LocalStore, MemoryStore

from cacheness.storage import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.lifecycle import AuthorityLifecycleEngine
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
from cacheness.storage.obstore_generation_io import ObstoreGenerationIO
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheConfigurationError,
    CacheUnsafePathError,
)


@dataclass(frozen=True)
class _McapRecord:
    """One value whose store-local handler owns an MCAP-style native file."""

    payload: bytes


class _McapHandler:
    """Representative custom handler that receives only private local paths."""

    data_type = "contract_mcap"
    payload_format = "mcap"
    payload_format_version = 1

    def __init__(self) -> None:
        self.put_paths: list[Path] = []
        self.staged_artifacts: list[Path] = []
        self.get_paths: list[Path] = []
        self.snapshot_metadata: list[dict[str, Any]] = []

    def can_handle(self, data: object, config: object = None) -> bool:
        del config
        return isinstance(data, _McapRecord)

    def put(
        self, data: _McapRecord, file_path: Path, config: object
    ) -> dict[str, object]:
        del config
        self.put_paths.append(file_path)
        artifact = file_path.with_suffix(".mcap")
        self.staged_artifacts.append(artifact)
        artifact.write_bytes(b"MCAP\x00" + data.payload)
        return {
            "actual_path": str(artifact),
            "file_size": artifact.stat().st_size,
            "payload_format": self.payload_format,
            "payload_format_version": self.payload_format_version,
            "metadata": {"native": "mcap"},
        }

    def get(self, file_path: Path, metadata: dict[str, Any]) -> _McapRecord:
        self.get_paths.append(file_path)
        self.snapshot_metadata.append(dict(metadata))
        assert file_path.suffix == ".mcap"
        assert metadata["actual_path"] == str(file_path)
        content = file_path.read_bytes()
        assert content.startswith(b"MCAP\x00")
        return _McapRecord(content.removeprefix(b"MCAP\x00"))

    def get_file_extension(self, config: object) -> str:
        del config
        return ".mcap"

    def supports_payload_contract(
        self, payload_format: str, payload_format_version: int
    ) -> bool:
        return (payload_format, payload_format_version) == ("mcap", 1)

    def payload_transformation_edges(self) -> tuple[object, ...]:
        return ()


class _RecordingStore:
    """Record exact SDK calls while forwarding to one real obstore store."""

    def __init__(self, store: object, *, allow_list: bool = False) -> None:
        self.store = store
        self.calls: list[tuple[str, str]] = []
        self.put_options: list[dict[str, object]] = []
        self.list_calls = 0
        self.allow_list = allow_list

    def put(self, locator: str, source: object, **kwargs: object) -> object:
        self.calls.append(("put", locator))
        self.put_options.append(dict(kwargs))
        return self.store.put(locator, source, **kwargs)

    def get(self, locator: str) -> object:
        self.calls.append(("get", locator))
        return self.store.get(locator)

    def head(self, locator: str) -> object:
        self.calls.append(("head", locator))
        return self.store.head(locator)

    def delete(self, locators: list[str]) -> object:
        self.calls.extend(("delete", locator) for locator in locators)
        return self.store.delete(locators)

    def list(self, *args: object, **kwargs: object) -> object:
        self.list_calls += 1
        if not self.allow_list:
            raise AssertionError("generation mechanics must not use listings as authority")
        return self.store.list(*args, **kwargs)


class _AcceptedThenRaisedStore(_RecordingStore):
    """Model one acknowledged create whose response is lost by the transport."""

    def put(self, locator: str, source: object, **kwargs: object) -> object:
        super().put(locator, source, **kwargs)
        raise OSError("simulated accepted create response loss")


class _MismatchedThenRaisedStore(_RecordingStore):
    """Model an ambiguous create that published different immutable bytes."""

    def put(self, locator: str, source: object, **kwargs: object) -> object:
        self.calls.append(("put", locator))
        original = source.read()
        self.store.put(locator, b"x" * len(original), **kwargs)
        raise OSError("simulated accepted create response loss")


class _PreconditionCollisionStore(_RecordingStore):
    """Expose the documented create-precondition collision variant."""

    def put(self, locator: str, source: object, **kwargs: object) -> object:
        del source, kwargs
        self.calls.append(("put", locator))
        raise PreconditionError("simulated create precondition")


class _DeterministicRemoteAuthority(InMemoryLifecycleAuthority):
    """Test-only remote-shaped authority for a mocked S3 transport contract."""

    qualification_identity = "postgresql"
    topology_capabilities = {
        "durable": True,
        "process_scope": "multi_host",
        "host_scope": "multi_host",
        "transaction_scope": "authority",
        "exact_cas": True,
        "portable_query": True,
        "canonical_scan": True,
        "index_acceleration": True,
    }


class _StaticRemoteManifestKey:
    """Provide deterministic shared test key material for the remote profile."""

    def get_key(self) -> bytes:
        return b"m" * 32

    def get_or_initialize_new_store(self) -> bytes:
        return self.get_key()

    def initialize_new_store(self) -> bytes:
        return self.get_key()


_S3_REGION = "us-east-1"
_S3_BUCKET = "cacheness-obstore-generation-contract"


@pytest.fixture
def moto_s3_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ObstoreGenerationIO, object]:
    """Build the sole test-only HTTP endpoint exception for mocked S3."""

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint_url = f"http://{host}:{port}"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", _S3_REGION)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=_S3_REGION,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    client.create_bucket(Bucket=_S3_BUCKET)
    handler_root = tmp_path / "handler-root"
    handler_root.mkdir()
    provider = ObstoreGenerationIO.for_s3(
        bucket=_S3_BUCKET,
        prefix="contract/s3",
        region=_S3_REGION,
        handler_root=handler_root,
        endpoint=endpoint_url,
        _allow_test_endpoint=True,
    )
    try:
        yield provider, client
    finally:
        provider.close()
        server.stop()


def _mocked_s3_blob_store(tmp_path: Path, provider: ObstoreGenerationIO) -> BlobStore:
    """Compose the real lifecycle engine with test-only remote-shaped authority."""

    return BlobStore(
        StoreTopology(
            payload=BackendRef(instance=provider),
            authority=BackendRef(instance=_DeterministicRemoteAuthority()),
        ),
        cache_dir=tmp_path / "mocked-s3-store",
        manifest_key_provider=_StaticRemoteManifestKey(),
    )


@pytest.fixture(params=("local", "memory"), ids=("local", "memory"))
def store_case(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[str, _RecordingStore, Path, BackendRef]:
    """Build one true-tier obstore participant without a second lifecycle seam."""

    name = str(request.param)
    handler_root = tmp_path / f"{name}-handler-root"
    handler_root.mkdir()
    if name == "local":
        return (
            name,
            _RecordingStore(LocalStore(tmp_path / "objects", mkdir=True)),
            handler_root,
            BackendRef(name="sqlite", options={"root": tmp_path / "authority"}),
        )
    return (
        name,
        _RecordingStore(MemoryStore()),
        handler_root,
        BackendRef(name="memory"),
    )


def _provider(
    store: object, handler_root: Path, *, identity: str, **limits: int
) -> ObstoreGenerationIO:
    """Construct the sole participant seam with explicit transfer limits."""

    return ObstoreGenerationIO(
        store,
        GuardedHandlerIO(handler_root),
        qualification_identity="filesystem" if identity == "local" else "memory",
        **limits,
    )


def test_local_store_provider_round_trips_native_npz_through_blob_store(
    tmp_path: Path,
) -> None:
    """A LocalStore participant keeps native NPZ handlers path-only and private."""
    payload_root = tmp_path / "payloads"
    provider = ObstoreGenerationIO(
        LocalStore(payload_root, mkdir=True),
        GuardedHandlerIO(payload_root),
        qualification_identity="filesystem",
    )
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(instance=provider),
            authority=BackendRef(name="sqlite", options={"root": tmp_path / "authority"}),
        ),
        cache_dir=tmp_path / "store",
    )
    value = np.asarray([1, 2, 3], dtype=np.int64)
    try:
        receipt = store.put_entry(value, key="local-npz")

        assert store.get(receipt.key).tolist() == [1, 2, 3]
    finally:
        store.close()


def test_local_and_memory_providers_round_trip_one_store_local_mcap_handler(
    store_case: tuple[str, _RecordingStore, Path, BackendRef], tmp_path: Path
) -> None:
    """Custom handlers retain only private suffix-preserving Path boundaries."""

    name, object_store, handler_root, authority = store_case
    provider = _provider(object_store, handler_root, identity=name)
    store = BlobStore(
        StoreTopology(payload=BackendRef(instance=provider), authority=authority),
        cache_dir=tmp_path / "cache",
    )
    handler = _McapHandler()
    store.handlers.register_handler(handler, priority=0)
    value = _McapRecord(b"custom path handler")

    try:
        receipt = store.put_entry(value, key=f"{name}-mcap")

        assert store.get(receipt.key) == value
        assert handler.put_paths and handler.get_paths
        assert all(path.suffix == ".mcap" for path in (*handler.staged_artifacts, *handler.get_paths))
        assert all(not path.is_relative_to(handler_root) for path in handler.put_paths)
        assert all(not path.is_relative_to(handler_root) for path in handler.get_paths)
        assert all(
            Path(metadata["actual_path"]).suffix == ".mcap"
            for metadata in handler.snapshot_metadata
        )
        assert object_store.list_calls == 0
    finally:
        store.close()


def test_local_and_memory_share_exact_immutable_generation_contract(
    store_case: tuple[str, _RecordingStore, Path, BackendRef]
) -> None:
    """Create, collision, exact read, deletion, and absence stay backend-neutral."""

    name, object_store, handler_root, _authority = store_case
    provider = _provider(object_store, handler_root, identity=name)
    handler = _McapHandler()
    locator = Path("generations") / "participant-contract" / "immutable.mcap"
    other_locator = Path("generations") / "participant-contract" / "other.mcap"

    try:
        with provider.stage(handler, _McapRecord(b"one"), config=None) as staged:
            published = provider.publish_generation(staged, locator)
        with provider.open_snapshot(locator, dict(published["metadata"])) as snapshot:
            assert snapshot.path.read_bytes() == b"MCAP\x00one"

        with provider.stage(handler, _McapRecord(b"two"), config=None) as staged:
            with pytest.raises(FileExistsError):
                provider.publish_generation(staged, locator)
        with provider.stage(handler, _McapRecord(b"other"), config=None) as staged:
            provider.publish_generation(staged, other_locator)

        provider.delete_or_prove_absent(locator)
        provider.delete_or_prove_absent(locator)
        with pytest.raises(FileNotFoundError):
            with provider.open_snapshot(locator, {}):
                pass
        with provider.open_snapshot(other_locator, {}) as snapshot:
            assert snapshot.path.read_bytes() == b"MCAP\x00other"
        assert object_store.list_calls == 0
    finally:
        provider.close()


@pytest.mark.parametrize("store_name", ("local", "memory"))
def test_accepted_then_lost_create_response_is_settled_by_exact_identity(
    tmp_path: Path, store_name: str
) -> None:
    """A matching exact object completes safely without using list discovery."""

    base_store = (
        LocalStore(tmp_path / "objects", mkdir=True)
        if store_name == "local"
        else MemoryStore()
    )
    wrapped = _AcceptedThenRaisedStore(base_store)
    handler_root = tmp_path / "handler-root"
    handler_root.mkdir()
    provider = _provider(wrapped, handler_root, identity=store_name)
    locator = Path("generations") / "response-loss" / "matching.mcap"

    try:
        with provider.stage(_McapHandler(), _McapRecord(b"matching"), config=None) as staged:
            published = provider.publish_generation(staged, locator)

        with provider.open_snapshot(locator, dict(published["metadata"])) as snapshot:
            assert snapshot.path.read_bytes() == b"MCAP\x00matching"
        assert ("head", locator.as_posix()) in wrapped.calls
        assert ("get", locator.as_posix()) in wrapped.calls
        assert wrapped.list_calls == 0
    finally:
        provider.close()


@pytest.mark.parametrize("store_name", ("local", "memory"))
def test_accepted_then_lost_create_response_rejects_mismatched_identity(
    tmp_path: Path, store_name: str
) -> None:
    """Ambiguity with different bytes is conflict, never an inferred success."""

    base_store = (
        LocalStore(tmp_path / "objects", mkdir=True)
        if store_name == "local"
        else MemoryStore()
    )
    wrapped = _MismatchedThenRaisedStore(base_store)
    handler_root = tmp_path / "handler-root"
    handler_root.mkdir()
    provider = _provider(wrapped, handler_root, identity=store_name)
    locator = Path("generations") / "response-loss" / "mismatch.mcap"

    try:
        with provider.stage(_McapHandler(), _McapRecord(b"matching"), config=None) as staged:
            with pytest.raises(CacheBlobLifecycleConflictError):
                provider.publish_generation(staged, locator)
        assert ("head", locator.as_posix()) in wrapped.calls
        assert ("get", locator.as_posix()) in wrapped.calls
        assert wrapped.list_calls == 0
    finally:
        provider.close()


def test_create_precondition_is_the_other_typed_immutable_collision(
    tmp_path: Path,
) -> None:
    """Only documented create preconditions normalize to the collision result."""

    wrapped = _PreconditionCollisionStore(MemoryStore())
    handler_root = tmp_path / "handler-root"
    handler_root.mkdir()
    provider = _provider(wrapped, handler_root, identity="memory")
    locator = Path("generations") / "response-loss" / "precondition.mcap"

    try:
        with provider.stage(_McapHandler(), _McapRecord(b"matching"), config=None) as staged:
            with pytest.raises(FileExistsError) as error_info:
                provider.publish_generation(staged, locator)
        assert isinstance(error_info.value.__cause__, PreconditionError)
        assert wrapped.list_calls == 0
    finally:
        provider.close()


@pytest.mark.parametrize(
    "locator",
    (
        "/absolute/path.mcap",
        "C:/drive/path.mcap",
        r"\\server\\share\\path.mcap",
        "s3://bucket/key.mcap",
        "generations/../escape.mcap",
        "alternate/participant/path.mcap",
        "generations/participant/path.mcap!",
        f"generations/participant/path.{('a' * 97)}",
    ),
)
def test_hostile_locator_is_rejected_before_any_obstore_call(
    tmp_path: Path, locator: str
) -> None:
    """Untrusted locator text cannot reach an object participant or its root."""

    wrapped = _RecordingStore(MemoryStore())
    handler_root = tmp_path / "handler-root"
    handler_root.mkdir()
    provider = _provider(wrapped, handler_root, identity="memory")
    try:
        with pytest.raises(CacheUnsafePathError):
            with provider.open_snapshot(locator, {}):
                pass
        assert wrapped.calls == []
        assert wrapped.list_calls == 0
    finally:
        provider.close()


def test_transfer_bounds_fail_before_unbounded_upload_or_download(
    tmp_path: Path,
) -> None:
    """Configured resource caps reject work before handler-facing I/O can grow."""

    wrapped = _RecordingStore(MemoryStore())
    handler_root = tmp_path / "handler-root"
    handler_root.mkdir()
    provider = _provider(
        wrapped,
        handler_root,
        identity="memory",
        max_upload_bytes=1,
        max_download_bytes=1,
    )
    locator = Path("generations") / "bounded" / "payload.mcap"

    try:
        with provider.stage(_McapHandler(), _McapRecord(b"bounded"), config=None) as staged:
            with pytest.raises(CacheBlobBackendError):
                provider.publish_generation(staged, locator)
        assert wrapped.calls == []
        wrapped.store.put(locator.as_posix(), b"MCAP\x00bounded", mode="create")
        with pytest.raises(CacheBlobBackendError):
            with provider.open_snapshot(locator, {}):
                pass
        assert wrapped.calls == [("get", locator.as_posix())]
        assert wrapped.list_calls == 0
    finally:
        provider.close()


def test_mocked_s3_provider_completes_one_authoritative_lifecycle_for_native_and_mcap(
    tmp_path: Path,
    moto_s3_provider: tuple[ObstoreGenerationIO, object],
) -> None:
    """Moto exercises object mechanics; the real engine still owns visibility."""

    provider, _client = moto_s3_provider
    store = _mocked_s3_blob_store(tmp_path, provider)
    handler = _McapHandler()
    store.handlers.register_handler(handler, priority=0)
    native_value = np.asarray([11, 13, 17], dtype=np.int64)
    custom_value = _McapRecord(b"mocked S3 custom handler")

    try:
        native_receipt = store.put_entry(native_value, key="s3-native")
        custom_receipt = store.put_entry(custom_value, key="s3-mcap")

        assert type(store.lifecycle) is AuthorityLifecycleEngine
        assert store.topology.qualified_profile.pair == ("postgresql", "s3")
        assert store.get(native_receipt.key).tolist() == [11, 13, 17]
        assert store.get(custom_receipt.key) == custom_value
        assert handler.put_paths and handler.get_paths
        assert all(path.suffix == ".mcap" for path in handler.get_paths)
        assert all(not path.is_relative_to(provider.root) for path in handler.get_paths)

        assert store.delete(native_receipt.key) is True
        assert store.delete(custom_receipt.key) is True
    finally:
        store.close()


def test_mocked_s3_collision_and_lost_create_response_settle_only_by_exact_object(
    moto_s3_provider: tuple[ObstoreGenerationIO, object],
) -> None:
    """One immutable S3 object wins; response loss never consults an inventory."""

    provider, _client = moto_s3_provider
    handler = _McapHandler()
    locator = Path("generations") / "mocked-s3" / "response-loss.mcap"

    try:
        with provider.stage(handler, _McapRecord(b"first"), config=None) as staged:
            published = provider.publish_generation(staged, locator)
        with provider.stage(handler, _McapRecord(b"second"), config=None) as staged:
            with pytest.raises(FileExistsError):
                provider.publish_generation(staged, locator)
        with provider.open_snapshot(locator, dict(published["metadata"])) as snapshot:
            assert snapshot.path.read_bytes() == b"MCAP\x00first"

        provider._store = _AcceptedThenRaisedStore(provider._store)
        lost_locator = Path("generations") / "mocked-s3" / "lost-response.mcap"
        with provider.stage(handler, _McapRecord(b"response loss"), config=None) as staged:
            recovered = provider.publish_generation(staged, lost_locator)
        with provider.open_snapshot(lost_locator, dict(recovered["metadata"])) as snapshot:
            assert snapshot.path.read_bytes() == b"MCAP\x00response loss"
        assert provider._store.list_calls == 0
    finally:
        provider.close()


@pytest.mark.parametrize(
    "invalid_options",
    (
        {"bucket": ""},
        {"bucket": "bucket/escape"},
        {"region": ""},
        {"prefix": "../escape"},
        {"endpoint": "https://objects.example.test"},
        {"endpoint": "http://127.0.0.1:not-a-port"},
        {"expected_bucket_owner": "123456789012"},
    ),
)
def test_s3_configuration_rejects_unsafe_or_unsupported_account_boundaries(
    tmp_path: Path, invalid_options: dict[str, str]
) -> None:
    """D-16 rejects owner pinning rather than rebuilding an unsafe workaround."""

    handler_root = tmp_path / "handler-root"
    handler_root.mkdir()
    options: dict[str, object] = {
        "bucket": _S3_BUCKET,
        "prefix": "contract/s3",
        "region": _S3_REGION,
        "handler_root": handler_root,
    }
    options.update(invalid_options)

    with pytest.raises(CacheConfigurationError):
        ObstoreGenerationIO.for_s3(**options)


def test_mocked_s3_enforces_direct_put_bounds_and_preserves_exact_maintenance_scope(
    moto_s3_provider: tuple[ObstoreGenerationIO, object],
) -> None:
    """S3 mechanics are bounded evidence, never a catalog or visibility authority."""

    provider, client = moto_s3_provider
    handler = _McapHandler()
    locator = Path("generations") / "mocked-s3" / "bounded.mcap"
    other_key = "contract/s3/generations/mock-scope/unrelated.mcap"
    client.put_object(Bucket=_S3_BUCKET, Key=other_key, Body=b"unrelated")
    wrapped = _RecordingStore(provider._store, allow_list=True)
    provider._store = wrapped
    provider.max_upload_bytes = 1

    try:
        with provider.stage(handler, _McapRecord(b"too large"), config=None) as staged:
            with pytest.raises(CacheBlobBackendError, match="upload bound"):
                provider.publish_generation(staged, locator)
        assert wrapped.calls == []

        provider.max_upload_bytes = 128 * 1024 * 1024
        with provider.stage(handler, _McapRecord(b"exact"), config=None) as staged:
            provider.publish_generation(staged, locator)
        assert wrapped.put_options == [{"mode": "create", "use_multipart": False}]
        actual_head = wrapped.head

        def opaque_head(_locator: str) -> dict[str, object]:
            return {
                "size": len(b"MCAP\x00exact"),
                "e_tag": '"multipart-looking-opaque-7"',
                "version": "opaque-version",
            }

        wrapped.head = opaque_head
        evidence = provider.head_generation(locator)
        assert evidence.byte_size == len(b"MCAP\x00exact")
        assert evidence.e_tag == '"multipart-looking-opaque-7"'
        assert evidence.version == "opaque-version"
        assert not hasattr(evidence, "digest")
        wrapped.head = actual_head

        page = provider.inventory_page(max_objects=1)
        assert page.objects
        assert page.next_offset is not None
        assert all(item.locator.startswith("generations/") for item in page.objects)

        provider.delete_or_prove_absent(locator)
        client.head_object(Bucket=_S3_BUCKET, Key=other_key)
        assert wrapped.list_calls == 1
    finally:
        provider.close()
