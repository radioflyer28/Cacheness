"""Tracer coverage for BlobStore's canonical committed read contract."""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobManifestMalformedError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobPayloadTamperedError,
    CacheReason,
)
from cacheness.metadata import InMemoryBackend
from cacheness.storage import BlobStore
from cacheness.storage import blob_store as blob_store_module
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.manifest import BlobManifestV1
from cacheness.storage.read_contract import (
    CacheReadFailureCategory,
    classify_cache_read_failure,
)
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


class _TracingHandler:
    """Small native handler that makes read ordering observable."""

    data_type = "tracing_object"

    def __init__(self, events: list[str]):
        self.events = events

    def put(self, data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
        payload_path = file_path.with_suffix(".trace")
        payload_path.write_text(str(data), encoding="utf-8")
        return {
            "storage_format": "trace",
            "file_size": payload_path.stat().st_size,
            "actual_path": str(payload_path),
            "metadata": {"storage_format": "trace"},
        }

    def get(self, file_path: Path, _metadata: dict[str, Any]) -> str:
        self.events.append("handler")
        return file_path.read_text(encoding="utf-8")


class _SingleHandlerRegistry:
    """Use one simple handler for the tracer's real storage path."""

    def __init__(self, handler: _TracingHandler):
        self.handler = handler

    def get_handler(self, _data: Any) -> _TracingHandler:
        return self.handler

    def get_handler_by_type(self, data_type: str) -> _TracingHandler:
        assert data_type == self.handler.data_type
        return self.handler


class _UnsupportedBackend(InMemoryBackend):
    """An inherited metadata backend that is not an exact canonical identity."""


def test_authority_tracer_put_read_and_reopen_uses_committed_authority(
    tmp_path: Path,
) -> None:
    """A direct store round-trip keeps native bytes outside authority writes."""
    from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority

    root = tmp_path / "authority-tracer"
    authority = SqliteLifecycleAuthority.for_root(root)
    store = BlobStore(root, lifecycle_authority=authority)
    events: list[str] = []
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    assert not authority.path.exists()
    assert store.put("native payload", key="tracer") == "tracer"
    assert authority.open_write_transactions == 0
    assert store.get("tracer") == "native payload"
    assert events == ["handler"]
    store.close()

    reopened_authority = SqliteLifecycleAuthority.for_root(root)
    reopened = BlobStore(root, lifecycle_authority=reopened_authority)
    reopened.handlers = _SingleHandlerRegistry(_TracingHandler([]))
    assert reopened.get("tracer") == "native payload"
    reopened.close()


def _authority_entry_with_replaced_manifest(
    store: BlobStore, key: str, **changes: Any
) -> object:
    """Return a signed authority snapshot altered only for a fail-closed test."""
    entry = store.lifecycle_authority.read_entry(key)
    assert entry is not None
    manifest = BlobManifestV1.from_canonical_bytes(entry.manifest)
    changed_manifest = store.lifecycle._sign(
        replace(manifest, **changes, signature="")
    )
    changed_bytes = changed_manifest.canonical_bytes()
    return replace(
        entry,
        manifest=changed_bytes,
        expectation=replace(
            entry.expectation,
            manifest_digest=hashlib.sha256(changed_bytes).hexdigest(),
        ),
    )


def _authority_entry_with_invalid_signature(store: BlobStore, key: str) -> object:
    """Return an otherwise valid authority snapshot with an invalid signature."""
    entry = store.lifecycle_authority.read_entry(key)
    assert entry is not None
    manifest = BlobManifestV1.from_canonical_bytes(entry.manifest)
    changed_bytes = replace(manifest, signature="0" * 64).canonical_bytes()
    return replace(
        entry,
        manifest=changed_bytes,
        expectation=replace(
            entry.expectation,
            manifest_digest=hashlib.sha256(changed_bytes).hexdigest(),
        ),
    )


def _authority_entry_with_raw_manifest(
    store: BlobStore, key: str, manifest: bytes
) -> object:
    """Return an authority snapshot with controlled raw manifest bytes."""
    entry = store.lifecycle_authority.read_entry(key)
    assert entry is not None
    return replace(
        entry,
        manifest=manifest,
        expectation=replace(
            entry.expectation,
            manifest_digest=hashlib.sha256(manifest).hexdigest(),
        ),
    )


@pytest.mark.parametrize("failure", ("malformed_legacy", "unsupported_backend"))
def test_constructor_failure_closes_managed_root_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Failed legacy recognition and backend selection release GuardedHandlerIO."""
    closed: list[GuardedHandlerIO] = []
    original_close = GuardedHandlerIO.close

    def close_spy(adapter: GuardedHandlerIO) -> None:
        closed.append(adapter)
        original_close(adapter)

    monkeypatch.setattr(GuardedHandlerIO, "close", close_spy)
    cache_dir = tmp_path / failure
    if failure == "malformed_legacy":
        cache_dir.mkdir()
        (cache_dir / "provenance.json").write_text("not json", encoding="utf-8")
        with pytest.raises(CacheBlobManifestMalformedError):
            BlobStore(cache_dir)
    else:
        cache_dir.mkdir()
        with pytest.raises(CacheBlobBackendError):
            BlobStore(cache_dir, backend=_UnsupportedBackend())

    assert len(closed) == 1


def test_failed_initialization_closes_only_internally_owned_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constructor failure releases its default authority and projection."""
    created_authorities: list[SqliteLifecycleAuthority] = []
    closed_authorities: list[SqliteLifecycleAuthority] = []
    created_backends: list[InMemoryBackend] = []
    closed_backends: list[InMemoryBackend] = []
    original_for_root = SqliteLifecycleAuthority.for_root
    original_authority_close = SqliteLifecycleAuthority.close
    original_backend_init = InMemoryBackend.__init__
    original_backend_close = InMemoryBackend.close

    def tracked_for_root(*args: Any, **kwargs: Any) -> SqliteLifecycleAuthority:
        authority = original_for_root(*args, **kwargs)
        created_authorities.append(authority)
        return authority

    def tracked_authority_close(authority: SqliteLifecycleAuthority) -> None:
        if authority in created_authorities:
            closed_authorities.append(authority)
        original_authority_close(authority)

    def tracked_backend_init(backend: InMemoryBackend) -> None:
        original_backend_init(backend)
        created_backends.append(backend)

    def tracked_backend_close(backend: InMemoryBackend) -> None:
        if backend in created_backends:
            closed_backends.append(backend)
        original_backend_close(backend)

    monkeypatch.setattr(SqliteLifecycleAuthority, "for_root", tracked_for_root)
    monkeypatch.setattr(SqliteLifecycleAuthority, "close", tracked_authority_close)
    monkeypatch.setattr(InMemoryBackend, "__init__", tracked_backend_init)
    monkeypatch.setattr(InMemoryBackend, "close", tracked_backend_close)
    monkeypatch.setattr(
        blob_store_module,
        "HandlerRegistry",
        lambda: (_ for _ in ()).throw(RuntimeError("handler setup failed")),
    )

    with pytest.raises(RuntimeError) as failure:
        BlobStore(tmp_path / "owned-authority")

    retained_failure = failure.value
    assert retained_failure.args == ("handler setup failed",)
    assert len(created_authorities) == 1
    assert closed_authorities == created_authorities
    assert len(created_backends) == 1
    assert closed_backends == created_backends


def test_failed_initialization_does_not_close_caller_injected_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Injected authority and backend remain owned by their caller on failure."""
    backend = InMemoryBackend()
    authority = SqliteLifecycleAuthority.for_root(tmp_path / "injected-backend")
    closed: list[InMemoryBackend] = []
    closed_authorities: list[SqliteLifecycleAuthority] = []
    original_close = InMemoryBackend.close
    original_authority_close = SqliteLifecycleAuthority.close

    def tracked_close(candidate: InMemoryBackend) -> None:
        if candidate is backend:
            closed.append(candidate)
        original_close(candidate)

    def tracked_authority_close(candidate: SqliteLifecycleAuthority) -> None:
        if candidate is authority:
            closed_authorities.append(candidate)
        original_authority_close(candidate)

    monkeypatch.setattr(InMemoryBackend, "close", tracked_close)
    monkeypatch.setattr(SqliteLifecycleAuthority, "close", tracked_authority_close)
    monkeypatch.setattr(
        blob_store_module,
        "HandlerRegistry",
        lambda: (_ for _ in ()).throw(RuntimeError("handler setup failed")),
    )

    with pytest.raises(RuntimeError, match="handler setup failed"):
        BlobStore(
            tmp_path / "injected-backend",
            backend=backend,
            lifecycle_authority=authority,
        )

    assert closed == []
    assert closed_authorities == []


@pytest.mark.parametrize("signal_type", (KeyboardInterrupt, SystemExit))
def test_constructor_cancellation_closes_owned_authority_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
) -> None:
    """Cancellation after authority setup releases every owned resource."""
    closed_io: list[GuardedHandlerIO] = []
    created_authorities: list[SqliteLifecycleAuthority] = []
    closed_authorities: list[SqliteLifecycleAuthority] = []
    created_backends: list[InMemoryBackend] = []
    closed_backends: list[InMemoryBackend] = []
    original_io_close = GuardedHandlerIO.close
    original_for_root = SqliteLifecycleAuthority.for_root
    original_authority_close = SqliteLifecycleAuthority.close
    original_backend_init = InMemoryBackend.__init__
    original_backend_close = InMemoryBackend.close
    cancellation = signal_type("constructor cancellation")

    def close_io_spy(adapter: GuardedHandlerIO) -> None:
        closed_io.append(adapter)
        original_io_close(adapter)

    def tracked_for_root(*args: Any, **kwargs: Any) -> SqliteLifecycleAuthority:
        authority = original_for_root(*args, **kwargs)
        created_authorities.append(authority)
        return authority

    def close_authority_spy(authority: SqliteLifecycleAuthority) -> None:
        if authority in created_authorities:
            closed_authorities.append(authority)
        original_authority_close(authority)

    def tracked_backend_init(backend: InMemoryBackend) -> None:
        original_backend_init(backend)
        created_backends.append(backend)

    def close_backend_spy(backend: InMemoryBackend) -> None:
        if backend in created_backends:
            closed_backends.append(backend)
        original_backend_close(backend)

    def interrupt_handler_setup() -> None:
        raise cancellation

    monkeypatch.setattr(GuardedHandlerIO, "close", close_io_spy)
    monkeypatch.setattr(SqliteLifecycleAuthority, "for_root", tracked_for_root)
    monkeypatch.setattr(SqliteLifecycleAuthority, "close", close_authority_spy)
    monkeypatch.setattr(InMemoryBackend, "__init__", tracked_backend_init)
    monkeypatch.setattr(InMemoryBackend, "close", close_backend_spy)
    monkeypatch.setattr(
        blob_store_module,
        "HandlerRegistry",
        interrupt_handler_setup,
    )

    cache_dir = tmp_path / "owned-authority-cancellation"
    cache_dir.mkdir()
    with pytest.raises(signal_type) as error:
        BlobStore(cache_dir)

    assert error.value is cancellation
    assert len(closed_io) == 1
    assert len(created_authorities) == 1
    assert closed_authorities == created_authorities
    assert len(created_backends) == 1
    assert closed_backends == created_backends


@pytest.mark.parametrize("signal_type", (KeyboardInterrupt, SystemExit))
def test_constructor_cancellation_before_backend_creation_closes_guarded_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
) -> None:
    """A cancellation before backend setup still closes the acquired root descriptor."""
    closed_io: list[GuardedHandlerIO] = []
    original_io_close = GuardedHandlerIO.close
    cancellation = signal_type("configuration cancellation")

    def close_io_spy(adapter: GuardedHandlerIO) -> None:
        closed_io.append(adapter)
        original_io_close(adapter)

    def interrupt_config(*_args: Any, **_kwargs: Any) -> None:
        raise cancellation

    monkeypatch.setattr(GuardedHandlerIO, "close", close_io_spy)
    monkeypatch.setattr(blob_store_module, "CacheConfig", interrupt_config)

    cache_dir = tmp_path / "before-backend"
    cache_dir.mkdir()
    with pytest.raises(signal_type) as error:
        BlobStore(cache_dir, backend="json")

    assert error.value is cancellation
    assert len(closed_io) == 1


@pytest.mark.parametrize("signal_type", (KeyboardInterrupt, SystemExit))
def test_constructor_cancellation_does_not_close_injected_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
) -> None:
    """Cancellation never transfers caller-injected backend ownership to BlobStore."""
    backend = InMemoryBackend()
    closed_io: list[GuardedHandlerIO] = []
    closed_backends: list[InMemoryBackend] = []
    original_io_close = GuardedHandlerIO.close
    original_backend_close = InMemoryBackend.close
    cancellation = signal_type("handler cancellation")

    def close_io_spy(adapter: GuardedHandlerIO) -> None:
        closed_io.append(adapter)
        original_io_close(adapter)

    def close_backend_spy(candidate: InMemoryBackend) -> None:
        if candidate is backend:
            closed_backends.append(candidate)
        original_backend_close(candidate)

    def interrupt_handler_setup() -> None:
        raise cancellation

    monkeypatch.setattr(GuardedHandlerIO, "close", close_io_spy)
    monkeypatch.setattr(InMemoryBackend, "close", close_backend_spy)
    monkeypatch.setattr(blob_store_module, "HandlerRegistry", interrupt_handler_setup)

    cache_dir = tmp_path / "injected"
    cache_dir.mkdir()
    with pytest.raises(signal_type) as error:
        BlobStore(cache_dir, backend=backend)

    assert error.value is cancellation
    assert len(closed_io) == 1
    assert closed_backends == []


def _call_direct_operation(store: BlobStore, operation: str, key: str) -> object:
    """Invoke one direct public API after its recovery admission boundary."""
    operations = {
        "get": lambda: store.get(key),
        "get_metadata": lambda: store.get_metadata(key),
        "exists": lambda: store.exists(key),
        "list": store.list,
        "put": lambda: store.put("admission payload", key=key),
        "update_metadata": lambda: store.update_metadata(key, {"tag": "value"}),
        "delete": lambda: store.delete(key),
        "clear": store.clear,
    }
    return operations[operation]()


@pytest.mark.parametrize(
    "operation",
    (
        "get",
        "get_metadata",
        "exists",
        "list",
        "put",
        "update_metadata",
        "delete",
        "clear",
    ),
)
def test_direct_operations_ignore_corrupt_json_projection(
    tmp_path: Path, operation: str
) -> None:
    """A corrupt compatibility projection cannot override authority state."""
    root = tmp_path / operation
    key = "admission-key"
    initial = BlobStore(root, backend="json")
    try:
        initial.put("stored", key=key)
    finally:
        initial.close()
    (root / "cache_metadata.json").write_bytes(b"{")

    store = BlobStore(root, backend="json")
    try:
        result = _call_direct_operation(store, operation, key)
        if operation == "get":
            assert result == "stored"
        elif operation == "get_metadata":
            assert isinstance(result, dict)
        elif operation == "exists":
            assert result is True
        elif operation == "list":
            assert result == [key]
        elif operation == "put":
            assert result == key
        elif operation == "update_metadata":
            assert result is True
        elif operation == "delete":
            assert result is True
        else:
            assert result == 1
        projection_path = root / "cache_metadata.json"
        if operation in {"get", "get_metadata", "exists", "list"}:
            assert projection_path.read_bytes() == b"{"
        else:
            projection = json.loads(projection_path.read_text(encoding="utf-8"))
            assert projection["_cacheness_authority_revision"] == (
                store.lifecycle_authority.snapshot_state().revision
            )
    finally:
        store.close()


def test_direct_reads_use_lifecycle_authority_without_predecessor_state(
    tmp_path: Path,
) -> None:
    """Direct reads use the authority without constructing predecessor state."""
    store = BlobStore(tmp_path / "authority-read", backend="json")
    try:
        assert store.lifecycle_authority is not None
        assert store.get("absent") is None
    finally:
        store.close()


def test_clear_translates_authority_snapshot_failure_without_mutating_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transactional clear-snapshot failure leaves committed state intact."""
    root = tmp_path / "authority-clear-snapshot"
    store = BlobStore(root, backend="json")
    try:
        key = store.put("preserved payload", key="preserved-key")
        entry_before = store.get_metadata(key)
        assert entry_before is not None
        authority_entry_before = store.lifecycle_authority.read_entry(key)
        assert authority_entry_before is not None
        payload_path = root / entry_before["metadata"]["actual_path"]
        payload_before = payload_path.read_bytes()
        failure = OSError("authority clear snapshot unavailable")

        monkeypatch.setattr(
            store.lifecycle_authority,
            "begin_clear",
            lambda: (_ for _ in ()).throw(failure),
        )

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert error.value.__cause__ is failure
        assert error.value.context["operation"] == "clear"
        assert (
            classify_cache_read_failure(error.value)
            is CacheReadFailureCategory.BACKEND_FAILURE
        )
        assert store.get_metadata(key) == entry_before
        assert store.lifecycle_authority.read_entry(key) == authority_entry_before
        assert payload_path.read_bytes() == payload_before
        assert store.get(key) == "preserved payload"
        assert not list(root.glob("tombstones/**/*"))
    finally:
        store.close()


def test_clear_translates_committed_tombstone_reclamation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authority tombstone cleanup is typed and reconciliation converges it."""
    root = tmp_path / "committed-reclamation"
    store = BlobStore(root, backend="json")
    try:
        key = store.put("committed payload", key="committed-key")
        entry = store.get_metadata(key)
        assert entry is not None
        payload_path = root / entry["metadata"]["actual_path"]
        failure = OSError("tombstone reclamation unavailable")
        failed = False

        def fail_one_cleanup(seam: str) -> None:
            nonlocal failed
            if not failed and seam == "cleanup.before_payload_delete":
                failed = True
                raise failure

        monkeypatch.setattr(store.lifecycle, "fault_hook", fail_one_cleanup)

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert failed
        assert error.value.context["operation"] == "clear"
        assert error.value.__cause__ is not None
        assert error.value.__cause__.__cause__ is failure
        assert (
            classify_cache_read_failure(error.value)
            is CacheReadFailureCategory.BACKEND_FAILURE
        )
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.reconcile(apply=True).applied
        assert reopened.get(key) is None
        assert not payload_path.exists()
    finally:
        reopened.close()


def test_clear_does_not_rewrap_already_typed_lifecycle_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direct boundary preserves an existing public BlobStore error object."""
    store = BlobStore(tmp_path / "pretyped-clear", backend="json")
    try:
        typed_failure = CacheBlobBackendError("already classified")

        def raise_typed_failure() -> int:
            raise typed_failure

        monkeypatch.setattr(store.lifecycle, "clear", raise_typed_failure)

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert error.value is typed_failure
    finally:
        store.close()


def test_tracer_direct_blob_store_reauthenticates_one_committed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read validates M1, snapshots, validates M2, then deserializes."""
    events: list[str] = []
    store = BlobStore(tmp_path / "tracer", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        key = store.put("tracer payload", key="tracer-key")

        entry = store.lifecycle_authority.read_entry(key)
        assert entry is not None
        manifest = json.loads(entry.manifest)
        assert manifest["schema_version"] == 1
        assert manifest["payload_format_version"] == 1
        assert manifest["state"] == "committed"
        assert manifest["key"] == key
        assert manifest["signature_algorithm"] == "hmac-sha256"
        assert manifest["signature"]

        original_read_entry = store.lifecycle_authority.read_entry

        def read_entry_with_event(blob_key: str):
            events.append("authority")
            return original_read_entry(blob_key)

        monkeypatch.setattr(
            store.lifecycle_authority,
            "read_entry",
            read_entry_with_event,
        )

        original_snapshot = store.guarded_handler_io.open_snapshot

        @contextmanager
        def snapshot_with_event(locator, metadata):
            events.append("snapshot")
            with original_snapshot(locator, metadata) as snapshot:
                yield snapshot

        monkeypatch.setattr(
            store.guarded_handler_io, "open_snapshot", snapshot_with_event
        )

        from cacheness.storage import lifecycle as lifecycle_module

        original_digest = lifecycle_module.sha256_and_size

        def digest_with_event(path):
            events.append("digest")
            return original_digest(path)

        monkeypatch.setattr(lifecycle_module, "sha256_and_size", digest_with_event)

        events.clear()
        assert store.get(key) == "tracer payload"
        assert events == ["authority", "snapshot", "authority", "digest", "handler"]
    finally:
        store.close()


def test_absent_authority_record_returns_none_without_snapshot_or_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Authority absence is the only direct-read outcome represented as None."""
    events: list[str] = []
    store = BlobStore(tmp_path / "absent", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        monkeypatch.setattr(
            GuardedHandlerIO,
            "open_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("an absent record must not open a snapshot")
            ),
        )
        monkeypatch.setattr(store.lifecycle_authority, "read_entry", lambda _key: None)
        assert store.get("absent-key") is None
        assert events == []
    finally:
        store.close()


def test_get_metadata_authenticates_committed_manifest_without_payload_io(
    tmp_path, monkeypatch
):
    """Metadata reads use signed canonical truth but never open a payload."""
    events: list[str] = []
    store = BlobStore(tmp_path / "metadata", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        key = store.put("metadata payload", key="metadata-key", metadata={"tag": "v1"})
        monkeypatch.setattr(
            store.guarded_handler_io,
            "open_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("get_metadata must not open a payload snapshot")
            ),
        )
        monkeypatch.setattr(
            store.backend,
            "update_access_time",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("get_metadata must not update access state")
            ),
        )

        metadata = store.get_metadata(key)

        assert metadata is not None
        assert metadata["cache_key"] == key
        assert metadata["data_type"] == "tracing_object"
        assert metadata["metadata"]["tag"] == "v1"
        assert events == []

        entry = store.lifecycle_authority.read_entry(key)
        assert entry is not None
        tampered = BlobManifestV1.from_canonical_bytes(entry.manifest)
        tampered_manifest = replace(tampered, signature="0" * 64).canonical_bytes()
        tampered_expectation = replace(
            entry.expectation,
            manifest_digest=hashlib.sha256(tampered_manifest).hexdigest(),
        )
        monkeypatch.setattr(
            store.lifecycle_authority,
            "read_entry",
            lambda _key: replace(
                entry,
                manifest=tampered_manifest,
                expectation=tampered_expectation,
            ),
        )

        with pytest.raises(CacheBlobManifestUnauthenticatedError):
            store.get_metadata(key)
        assert events == []
    finally:
        store.close()


def test_get_metadata_preserves_the_frozen_legacy_v1_dictionary_shape(tmp_path):
    """Authority work must leave direct metadata dictionaries structurally intact."""
    store = BlobStore(tmp_path / "frozen-metadata", backend="json")
    try:
        key = store.put("metadata shape", key="metadata-shape")

        metadata = store.get_metadata(key)

        assert metadata is not None
        assert set(metadata) == {
            "cache_key",
            "created_at",
            "data_type",
            "file_size",
            "metadata",
        }
        assert metadata["cache_key"] == key
        assert set(metadata["metadata"]) == {
            "actual_path",
            "compression_codec",
            "object_type",
            "serializer",
            "storage_format",
        }
    finally:
        store.close()


def test_exists_verifies_one_snapshot_without_deserializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existence means a valid authenticated manifest and intact payload."""
    events: list[str] = []
    store = BlobStore(tmp_path / "exists", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        key = store.put("exists payload", key="exists-key")
        original_snapshot = store.guarded_handler_io.open_snapshot

        @contextmanager
        def snapshot_with_event(locator, metadata):
            events.append("snapshot")
            with original_snapshot(locator, metadata) as snapshot:
                yield snapshot

        monkeypatch.setattr(
            store.guarded_handler_io, "open_snapshot", snapshot_with_event
        )

        from cacheness.storage import lifecycle as lifecycle_module

        original_digest = lifecycle_module.sha256_and_size

        def digest_with_event(path):
            events.append("digest")
            return original_digest(path)

        monkeypatch.setattr(lifecycle_module, "sha256_and_size", digest_with_event)

        assert store.exists(key) is True
        assert events == ["snapshot", "digest"]

        entry = store.lifecycle_authority.read_entry(key)
        assert entry is not None
        manifest = BlobManifestV1.from_canonical_bytes(entry.manifest)
        (store.cache_dir / manifest.locator).write_text(
            "tamper! payload", encoding="utf-8"
        )
        events.clear()

        with pytest.raises(CacheBlobPayloadTamperedError):
            store.exists(key)
        assert events == ["snapshot", "digest"]
        assert "handler" not in events
    finally:
        store.close()


def test_list_rejects_a_nonterminal_authority_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-terminal authority snapshot cannot be silently omitted."""
    events: list[str] = []
    store = BlobStore(tmp_path / "list", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        first = store.put("first", key="selected-first", metadata={"group": "one"})
        second = store.put("second", key="selected-second", metadata={"group": "one"})
        entry = store.lifecycle_authority.read_entry(second)
        assert entry is not None
        manifest = BlobManifestV1.from_canonical_bytes(entry.manifest)
        prepared = store.lifecycle._sign(
            replace(manifest, state="prepared", signature="")
        )
        prepared_bytes = prepared.canonical_bytes()
        prepared_entry = replace(
            entry,
            manifest=prepared_bytes,
            expectation=replace(
                entry.expectation,
                manifest_digest=hashlib.sha256(prepared_bytes).hexdigest(),
            ),
        )
        original_list_entries = store.lifecycle_authority.list_entries
        monkeypatch.setattr(
            store.lifecycle_authority,
            "list_entries",
            lambda: tuple(
                prepared_entry if candidate.key == second else candidate
                for candidate in original_list_entries()
            ),
        )

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.list(prefix="selected-", metadata_filter={"group": "one"})

        assert events == []
        assert first == "selected-first"
    finally:
        store.close()


def test_list_uses_one_authority_enumeration_without_legacy_reselection(
    tmp_path: Path,
) -> None:
    """Listing reads the selected authority directly without a legacy seam."""
    store = BlobStore(tmp_path / "list-disappeared", backend="json")
    try:
        store.put("payload", key="selected-key")
        assert store.list() == ["selected-key"]
    finally:
        store.close()


def test_update_metadata_resigns_only_user_metadata_and_rejects_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Public metadata patches cannot alter signed storage structure."""
    store = BlobStore(tmp_path / "update", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler([]))

    try:
        key = store.put("update payload", key="update-key", metadata={"owner": "one"})
        before_entry = store.lifecycle_authority.read_entry(key)
        assert before_entry is not None
        before = BlobManifestV1.from_canonical_bytes(before_entry.manifest)

        assert store.update_metadata(key, {"owner": "two", "label": "current"})

        after_entry = store.lifecycle_authority.read_entry(key)
        assert after_entry is not None
        after = BlobManifestV1.from_canonical_bytes(after_entry.manifest)
        assert dict(after.user_metadata) == {"owner": "two", "label": "current"}
        assert after.signature != before.signature
        assert after.generation != before.generation
        for field in (
            "schema_version",
            "key",
            "state",
            "locator",
            "handler_type",
            "payload_format",
            "payload_format_version",
            "digest_algorithm",
            "digest",
            "byte_size",
            "created_at",
            "handler_metadata",
            "signature_algorithm",
        ):
            assert getattr(after, field) == getattr(before, field)

        monkeypatch.setattr(
            store.lifecycle_authority,
            "prepare_mutation",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("structural patch must fail before authority prepare")
            ),
        )

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.update_metadata(key, {"locator": "/unsafe-replacement"})
        assert store.lifecycle_authority.read_entry(key) == after_entry
        assert store.update_metadata("absent-update", {"owner": "none"}) is False
    finally:
        store.close()


def test_update_metadata_rejects_a_stale_independent_store_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent same-generation patches have one CAS winner, never last write wins."""
    root = tmp_path / "independent-metadata-patch"
    first_store = BlobStore(root, backend="json")
    first_store.handlers = _SingleHandlerRegistry(_TracingHandler([]))
    second_store: BlobStore | None = None
    try:
        key = first_store.put("payload", key="patch-key", metadata={"owner": "base"})
        second_store = BlobStore(root, backend="json")
        publish_entered = Event()
        release_first_publish = Event()
        first_result: list[BaseException | bool] = []
        original_promote = first_store.lifecycle_authority.promote_mutation

        def block_first_promotion(*args: Any, **kwargs: Any) -> object:
            publish_entered.set()
            assert release_first_publish.wait(timeout=2)
            return original_promote(*args, **kwargs)

        monkeypatch.setattr(
            first_store.lifecycle_authority,
            "promote_mutation",
            block_first_promotion,
        )

        def patch_first_store() -> None:
            try:
                first_result.append(first_store.update_metadata(key, {"owner": "first"}))
            except BaseException as exc:
                first_result.append(exc)

        first_thread = Thread(target=patch_first_store)
        first_thread.start()
        assert publish_entered.wait(timeout=2)

        assert second_store.update_metadata(key, {"owner": "second"})
        release_first_publish.set()
        first_thread.join(timeout=2)

        assert not first_thread.is_alive()
        assert len(first_result) == 1
        assert isinstance(first_result[0], CacheBlobLifecycleConflictError)
        assert second_store.get_metadata(key)["metadata"]["owner"] == "second"
    finally:
        if second_store is not None:
            second_store.close()
        first_store.close()


def test_update_metadata_rejects_a_forged_authority_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A forged signed snapshot cannot satisfy the current authority CAS."""
    store = BlobStore(tmp_path / "outside-locator", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler([]))
    try:
        key = store.put("payload", key="outside-key")
        original_read_entry = store.lifecycle_authority.read_entry
        before = original_read_entry(key)
        assert before is not None
        outside_entry = _authority_entry_with_replaced_manifest(
            store, key, locator=str(tmp_path / "outside.bin")
        )

        monkeypatch.setattr(
            store.lifecycle_authority,
            "read_entry",
            lambda _key: outside_entry,
        )

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.update_metadata(key, {"label": "blocked"})

        assert original_read_entry(key) == before
    finally:
        store.close()


def test_delete_and_clear_reject_unauthenticated_authority_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct mutation does not reclaim payloads from unauthenticated state."""
    store = BlobStore(tmp_path / "mutate", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler([]))

    try:
        delete_key = store.put("delete payload", key="delete-key")
        original_read_entry = store.lifecycle_authority.read_entry
        delete_entry = original_read_entry(delete_key)
        assert delete_entry is not None
        delete_manifest = BlobManifestV1.from_canonical_bytes(delete_entry.manifest)
        payload_locator = store.cache_dir / delete_manifest.locator
        tampered_delete_entry = _authority_entry_with_invalid_signature(
            store, delete_key
        )
        original_delete = store.guarded_handler_io.file_ops.delete
        deleted_locators: list[Path] = []

        def reject_only_tampered_payload(locator: Path | str) -> bool:
            if Path(locator) in {Path(delete_manifest.locator), payload_locator}:
                raise AssertionError("unauthenticated manifest must not delete payload")
            deleted_locators.append(Path(locator))
            return original_delete(locator)

        with monkeypatch.context() as scoped:
            scoped.setattr(
                store.lifecycle_authority,
                "read_entry",
                lambda key: (
                    tampered_delete_entry
                    if key == delete_key
                    else original_read_entry(key)
                ),
            )
            scoped.setattr(
                store.guarded_handler_io.file_ops,
                "delete",
                reject_only_tampered_payload,
            )
            with pytest.raises(CacheBlobManifestUnauthenticatedError):
                store.delete(delete_key)
        assert deleted_locators == []
        assert original_read_entry(delete_key) == delete_entry

        first = store.put("first clear", key="clear-first")
        store.put("second clear", key="clear-second")
        tampered_first_entry = _authority_entry_with_invalid_signature(store, first)
        with monkeypatch.context() as scoped:
            scoped.setattr(
                store.lifecycle_authority,
                "read_entry",
                lambda key: (
                    tampered_first_entry
                    if key == first
                    else original_read_entry(key)
                ),
            )
            store.clear()
        assert original_read_entry(first) is not None
        assert store.get(first) == "first clear"
        assert store.delete("absent-delete") is False
    finally:
        store.close()


@pytest.mark.parametrize(
    ("fault", "error_type"),
    (
        ("malformed", CacheBlobManifestMalformedError),
        ("future_schema", CacheBlobManifestUnsupportedVersionError),
        ("nonterminal", CacheBlobLifecycleConflictError),
    ),
)
def test_every_direct_read_surface_preserves_ordered_typed_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    error_type: type[Exception],
):
    """Present corrupt, future, and conflicted records never become misses."""
    store = BlobStore(tmp_path / fault, backend="json")
    try:
        key = store.put({"fault": fault}, key="fault-key")
        entry = store.lifecycle_authority.read_entry(key)
        assert entry is not None
        manifest = BlobManifestV1.from_canonical_bytes(entry.manifest)
        payload_path = store.cache_dir / manifest.locator
        payload_before = payload_path.read_bytes()
        payload_mtime_before = payload_path.stat().st_mtime_ns
        if fault == "malformed":
            faulty_entry = _authority_entry_with_raw_manifest(store, key, b"{")
        elif fault == "future_schema":
            future_manifest = json.loads(entry.manifest)
            future_manifest["schema_version"] = 2
            faulty_entry = _authority_entry_with_raw_manifest(
                store,
                key,
                json.dumps(
                    future_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
        else:
            faulty_entry = _authority_entry_with_replaced_manifest(
                store, key, state="prepared"
            )
        original_read_entry = store.lifecycle_authority.read_entry
        original_list_entries = store.lifecycle_authority.list_entries
        monkeypatch.setattr(
            store.lifecycle_authority,
            "read_entry",
            lambda requested_key: (
                faulty_entry if requested_key == key else original_read_entry(requested_key)
            ),
        )
        monkeypatch.setattr(
            store.lifecycle_authority,
            "list_entries",
            lambda: tuple(
                faulty_entry if candidate.key == key else candidate
                for candidate in original_list_entries()
            ),
        )

        for operation in (
            lambda: store.get(key),
            lambda: store.get_metadata(key),
            lambda: store.exists(key),
            lambda: store.list(),
        ):
            with pytest.raises(error_type):
                operation()
            assert original_read_entry(key) == entry
            assert payload_path.read_bytes() == payload_before
            assert payload_path.stat().st_mtime_ns == payload_mtime_before
    finally:
        store.close()


def test_every_direct_read_surface_propagates_an_authority_backend_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """An authority backend fault stays typed across each direct read surface."""
    store = BlobStore(tmp_path / "backend-failure", backend="json")
    try:
        key = store.put({"backend": "failure"}, key="backend-key")
        failure = CacheBlobBackendError("injected lifecycle authority failure")
        monkeypatch.setattr(
            store.lifecycle_authority,
            "read_entry",
            lambda _key: (_ for _ in ()).throw(failure),
        )
        monkeypatch.setattr(
            store.lifecycle_authority,
            "list_entries",
            lambda: (_ for _ in ()).throw(failure),
        )

        for operation in (
            lambda: store.get(key),
            lambda: store.get_metadata(key),
            lambda: store.exists(key),
            lambda: store.list(),
        ):
            with pytest.raises(CacheBlobBackendError) as error:
                operation()
            assert error.value is failure
    finally:
        store.close()


@pytest.mark.parametrize("operation", ("list", "clear"))
def test_projection_backend_failures_do_not_block_authority_operations(
    tmp_path, monkeypatch: pytest.MonkeyPatch, operation: str
):
    """A compatibility projection failure cannot block authority-backed operations."""
    store = BlobStore(tmp_path / operation, backend="json")
    try:
        key = store.put({"operation": operation}, key="projection-key")
        projection_accessed = False

        def projection_unavailable(*_args, **_kwargs):
            nonlocal projection_accessed
            projection_accessed = True
            raise OSError("metadata projection unavailable")

        if hasattr(store.backend, "list_entries"):
            monkeypatch.setattr(store.backend, "list_entries", projection_unavailable)
        else:
            monkeypatch.setattr(
                store.backend,
                "get_entry",
                projection_unavailable,
            )

        result = getattr(store, operation)()
        assert result == ([key] if operation == "list" else 1)
        assert not projection_accessed
    finally:
        store.close()


# =============================================================================
# Authority composition and capability honesty (Plan 03-06)
# =============================================================================


def test_memory_authority_requires_explicit_ephemeral_topology_before_staging(
    tmp_path: Path,
) -> None:
    """Memory cannot silently satisfy the default durable multiprocess request."""
    from cacheness.config import CacheConfig, LifecycleAuthorityTopology
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority

    root = tmp_path / "memory-authority"
    with pytest.raises(CacheBlobBackendError) as error:
        BlobStore(root, backend="memory")

    assert (
        error.value.context["reason"]
        == CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED.value
    )
    assert not root.exists()

    ephemeral = CacheConfig(
        lifecycle_topology=LifecycleAuthorityTopology(
            durable=False,
            multiprocess=False,
            projection=False,
        )
    )
    store = BlobStore(root, backend="memory", config=ephemeral)
    try:
        assert type(store.lifecycle_authority) is InMemoryLifecycleAuthority
        assert store.lifecycle_authority.capabilities.durable is False
        assert store.lifecycle_authority.capabilities.multiprocess is False
        assert store.put("memory payload", key="memory-key") == "memory-key"
        assert store.get("memory-key") == "memory payload"
    finally:
        store.close()


def test_json_backend_is_a_rebuildable_authority_projection(tmp_path: Path) -> None:
    """Public JSON metadata is derived from SQLite authority, never read as truth."""
    from cacheness.metadata import JsonBackend

    root = tmp_path / "json-projection"
    store = BlobStore(root, backend="json")
    try:
        key = store.put("projected", key="projected-key", metadata={"source": "test"})

        assert type(store.backend) is JsonBackend
        projection = json.loads((root / "cache_metadata.json").read_text(encoding="utf-8"))
        assert projection["entries"][key] == store.get_metadata(key)
        assert projection["_cacheness_authority_revision"] == (
            store.lifecycle_authority.snapshot_state().revision
        )
    finally:
        store.close()
