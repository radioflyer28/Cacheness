"""Tracer coverage for BlobStore's canonical committed read contract."""

from contextlib import contextmanager
from dataclasses import replace
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
    CacheStorageError,
    CacheUnsafePathError,
)
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend
from cacheness.storage import BlobStore
from cacheness.storage import blob_store as blob_store_module
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.integrity import sign_hmac_sha256
from cacheness.storage.manifest import BlobManifestV1
from cacheness.storage.read_contract import (
    CacheReadFailureCategory,
    classify_cache_read_failure,
)


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


def _replace_signed_manifest(
    store: BlobStore, key: str, **changes: Any
) -> BlobManifestV1:
    """Replace one test manifest while preserving its canonical signature."""
    raw_manifest = store.manifest_repository.get_raw(key)
    assert raw_manifest is not None
    manifest = BlobManifestV1.from_canonical_bytes(raw_manifest)
    changed_manifest = replace(manifest, **changes)
    signed_manifest = changed_manifest.with_signature(
        sign_hmac_sha256(changed_manifest.signing_bytes(), store._manifest_key())
    )
    store.manifest_repository.put_raw(key, signed_manifest.canonical_bytes())
    return signed_manifest


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
        with pytest.raises(CacheBlobBackendError):
            BlobStore(cache_dir, backend=_UnsupportedBackend())

    assert len(closed) == 1


def test_failed_initialization_closes_only_internally_owned_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained constructor exception cannot defer owned SQLite cleanup to GC."""
    created: list[SqliteBackend] = []
    closed: list[SqliteBackend] = []
    original_init = SqliteBackend.__init__
    original_close = SqliteBackend.close

    def tracked_init(backend: SqliteBackend, *args: Any, **kwargs: Any) -> None:
        original_init(backend, *args, **kwargs)
        created.append(backend)

    def tracked_close(backend: SqliteBackend) -> None:
        if backend in created:
            closed.append(backend)
        original_close(backend)

    monkeypatch.setattr(SqliteBackend, "__init__", tracked_init)
    monkeypatch.setattr(SqliteBackend, "close", tracked_close)
    def fail_repository_setup(_backend: object, *, lifecycle_limits: object) -> None:
        assert lifecycle_limits is not None
        raise RuntimeError("repository setup failed")

    monkeypatch.setattr(
        blob_store_module,
        "create_manifest_repository",
        fail_repository_setup,
    )

    with pytest.raises(RuntimeError) as failure:
        BlobStore(tmp_path / "owned-sqlite", backend="sqlite")

    retained_failure = failure.value
    assert retained_failure.args == ("repository setup failed",)
    assert len(created) == 1
    assert closed == created


def test_failed_initialization_does_not_close_caller_injected_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An injected backend remains owned by its caller when setup fails later."""
    backend = InMemoryBackend()
    closed: list[InMemoryBackend] = []
    original_close = InMemoryBackend.close

    def tracked_close(candidate: InMemoryBackend) -> None:
        if candidate is backend:
            closed.append(candidate)
        original_close(candidate)

    monkeypatch.setattr(InMemoryBackend, "close", tracked_close)
    monkeypatch.setattr(
        blob_store_module,
        "HandlerRegistry",
        lambda: (_ for _ in ()).throw(RuntimeError("handler setup failed")),
    )

    with pytest.raises(RuntimeError, match="handler setup failed"):
        BlobStore(tmp_path / "injected-backend", backend=backend)

    assert closed == []


@pytest.mark.parametrize("signal_type", (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize(
    ("backend_name", "backend_type"),
    (("json", JsonBackend), ("sqlite", SqliteBackend)),
)
def test_constructor_cancellation_closes_owned_resources_after_backend_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
    backend_name: str,
    backend_type: type[JsonBackend] | type[SqliteBackend],
) -> None:
    """Cancellation after local backend construction releases every owned resource."""
    closed_io: list[GuardedHandlerIO] = []
    closed_backends: list[JsonBackend | SqliteBackend] = []
    initialized_backends: list[object] = []
    original_io_close = GuardedHandlerIO.close
    original_backend_close = backend_type.close
    cancellation = signal_type("constructor cancellation")

    def close_io_spy(adapter: GuardedHandlerIO) -> None:
        closed_io.append(adapter)
        original_io_close(adapter)

    def close_backend_spy(backend: JsonBackend | SqliteBackend) -> None:
        closed_backends.append(backend)
        original_backend_close(backend)

    def interrupt_repository_setup(
        backend: object, *, lifecycle_limits: object
    ) -> None:
        initialized_backends.append(backend)
        assert lifecycle_limits is not None
        raise cancellation

    monkeypatch.setattr(GuardedHandlerIO, "close", close_io_spy)
    monkeypatch.setattr(backend_type, "close", close_backend_spy)
    monkeypatch.setattr(
        blob_store_module,
        "create_manifest_repository",
        interrupt_repository_setup,
    )

    with pytest.raises(signal_type) as error:
        BlobStore(tmp_path / f"owned-{backend_name}", backend=backend_name)

    assert error.value is cancellation
    assert len(closed_io) == 1
    assert len(initialized_backends) == 1
    assert closed_backends.count(initialized_backends[0]) == 1


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

    with pytest.raises(signal_type) as error:
        BlobStore(tmp_path / "before-backend", backend="json")

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

    with pytest.raises(signal_type) as error:
        BlobStore(tmp_path / "injected", backend=backend)

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
    ("operation", "is_read"),
    (
        ("get", True),
        ("get_metadata", True),
        ("exists", True),
        ("list", True),
        ("put", False),
        ("update_metadata", False),
        ("delete", False),
        ("clear", False),
    ),
)
def test_all_direct_operations_translate_json_admission_refresh_failures(
    tmp_path: Path, operation: str, is_read: bool
) -> None:
    """Corrupt admission-time JSON state is a typed backend failure everywhere."""
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
        with pytest.raises(CacheBlobBackendError) as error:
            _call_direct_operation(store, operation, key)
        assert isinstance(error.value.__cause__, CacheStorageError)
        if is_read:
            assert (
                classify_cache_read_failure(error.value)
                is CacheReadFailureCategory.BACKEND_FAILURE
            )
    finally:
        store.close()


@pytest.mark.parametrize("state", ("active", "prepared", "poisoned"))
def test_recovery_conflicts_translate_to_lifecycle_failures(
    tmp_path: Path, state: str
) -> None:
    """Active, prepared, and poisoned recovery evidence blocks direct reads stably."""
    store = BlobStore(tmp_path / state, backend="json")
    coordinator = store._clear_recovery
    assert coordinator is not None
    try:
        if state == "active":
            with coordinator.admission(blocking=True):
                with pytest.raises(CacheBlobLifecycleConflictError) as error:
                    store.get("blocked")
        elif state == "prepared":
            coordinator._create_journal(coordinator._new_prepared_journal([]))
            with pytest.raises(CacheBlobLifecycleConflictError) as error:
                store.get("blocked")
        else:
            coordinator._poisoned = True
            with pytest.raises(CacheBlobLifecycleConflictError) as error:
                store.get("blocked")

        assert isinstance(error.value.__cause__, CacheStorageError)
        assert (
            classify_cache_read_failure(error.value)
            is CacheReadFailureCategory.LIFECYCLE_CONFLICT
        )
    finally:
        store.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite", "memory"))
@pytest.mark.parametrize("failure_site", ("stage", "backend_clear"))
def test_clear_translates_rolled_back_operational_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    failure_site: str,
) -> None:
    """Rolled-back clear faults stay typed at every supported local API boundary."""
    root = tmp_path / f"clear-{backend_name}-{failure_site}"
    backend = InMemoryBackend() if backend_name == "memory" else backend_name
    store = BlobStore(root, backend=backend)
    try:
        key = store.put("preserved payload", key="preserved-key")
        entry_before = store.get_metadata(key)
        assert entry_before is not None
        payload_path = Path(entry_before["metadata"]["actual_path"])
        payload_before = payload_path.read_bytes()
        coordinator = store._clear_recovery
        assert coordinator is not None
        failure = OSError(f"{failure_site} unavailable")

        if failure_site == "stage":

            def fail_staging(_mapping: dict[str, Any]) -> None:
                raise failure

            monkeypatch.setattr(coordinator, "_stage_mapping", fail_staging)
        else:

            def fail_backend_clear() -> int:
                raise failure

            monkeypatch.setattr(store.backend, "clear_all", fail_backend_clear)

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert error.value.__cause__ is failure
        assert error.value.context["operation"] == "clear"
        assert (
            classify_cache_read_failure(error.value)
            is CacheReadFailureCategory.BACKEND_FAILURE
        )
        assert store.get_metadata(key) == entry_before
        assert payload_path.read_bytes() == payload_before
        assert store.get(key) == "preserved payload"
        assert not coordinator.journal_path.exists()
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        store.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_clear_translates_committed_tombstone_reclamation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """Committed cleanup errors are typed while reopening converges clear authority."""
    root = tmp_path / f"committed-reclamation-{backend_name}"
    store = BlobStore(root, backend=backend_name)
    try:
        key = store.put("committed payload", key="committed-key")
        entry = store.get_metadata(key)
        assert entry is not None
        payload_path = Path(entry["metadata"]["actual_path"])
        coordinator = store._clear_recovery
        assert coordinator is not None
        delete_durable = store.guarded_handler_io.file_ops.delete_durable
        failure = OSError("tombstone reclamation unavailable")
        failed = False

        def fail_one_tombstone_reclamation(locator: Path | str) -> bool:
            nonlocal failed
            if not failed and Path(locator).name.startswith("clear-tombstone-"):
                failed = True
                raise failure
            return delete_durable(locator)

        monkeypatch.setattr(
            store.guarded_handler_io.file_ops,
            "delete_durable",
            fail_one_tombstone_reclamation,
        )

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert failed
        assert error.value.__cause__ is failure
        assert (
            classify_cache_read_failure(error.value)
            is CacheReadFailureCategory.BACKEND_FAILURE
        )
        assert coordinator.journal_path.exists()
    finally:
        store.close()

    reopened = BlobStore(root, backend=backend_name)
    try:
        assert reopened.get(key) is None
        assert not payload_path.exists()
        assert not reopened._clear_recovery.journal_path.exists()
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        reopened.close()


def test_clear_does_not_rewrap_already_typed_coordinator_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direct boundary preserves an existing public BlobStore error object."""
    store = BlobStore(tmp_path / "pretyped-clear", backend="json")
    try:
        coordinator = store._clear_recovery
        assert coordinator is not None
        typed_failure = CacheBlobBackendError("already classified")

        def raise_typed_failure(_mappings: list[tuple[str, Path]]) -> int:
            raise typed_failure

        monkeypatch.setattr(coordinator, "clear", raise_typed_failure)

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert error.value is typed_failure
    finally:
        store.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_tracer_direct_blob_store_reauthenticates_one_committed_snapshot(
    tmp_path, monkeypatch, backend_name
):
    """A read validates M1, snapshots, validates M2, then deserializes."""
    events: list[str] = []
    store = BlobStore(tmp_path / "tracer", backend=backend_name)
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        key = store.put("tracer payload", key="tracer-key")

        raw_record = store.manifest_repository.get_raw(key)
        assert raw_record is not None
        manifest = json.loads(raw_record)
        assert manifest["schema_version"] == 1
        assert manifest["payload_format_version"] == 1
        assert manifest["state"] == "committed"
        assert manifest["key"] == key
        assert manifest["signature_algorithm"] == "hmac-sha256"
        assert manifest["signature"]

        original_get_raw = store.manifest_repository.get_raw

        def get_raw_with_event(blob_key: str):
            events.append("repository")
            return original_get_raw(blob_key)

        monkeypatch.setattr(store.manifest_repository, "get_raw", get_raw_with_event)

        original_snapshot = store.guarded_handler_io.open_snapshot

        @contextmanager
        def snapshot_with_event(locator, metadata):
            events.append("snapshot")
            with original_snapshot(locator, metadata) as snapshot:
                yield snapshot

        monkeypatch.setattr(
            store.guarded_handler_io, "open_snapshot", snapshot_with_event
        )

        from cacheness.storage import blob_store as blob_store_module

        original_digest = blob_store_module.sha256_and_size

        def digest_with_event(path):
            events.append("digest")
            return original_digest(path)

        monkeypatch.setattr(blob_store_module, "sha256_and_size", digest_with_event)

        events.clear()
        assert store.get(key) == "tracer payload"
        assert events == ["repository", "snapshot", "repository", "digest", "handler"]
    finally:
        store.close()


def test_absent_raw_record_returns_none_without_snapshot_or_handler(tmp_path, monkeypatch):
    """Repository absence is the only direct-read outcome represented as None."""
    events: list[str] = []
    store = BlobStore(tmp_path / "absent", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        monkeypatch.setattr(
            store.guarded_handler_io,
            "open_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("an absent record must not open a snapshot")
            ),
        )
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

        raw_manifest = store.manifest_repository.get_raw(key)
        assert raw_manifest is not None
        tampered = json.loads(raw_manifest)
        tampered["signature"] = "0" * 64
        store.manifest_repository.put_raw(
            key,
            json.dumps(
                tampered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
        )

        with pytest.raises(CacheBlobManifestUnauthenticatedError):
            store.get_metadata(key)
        assert events == []
    finally:
        store.close()


def test_exists_verifies_one_snapshot_without_deserializing(tmp_path, monkeypatch):
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

        from cacheness.storage import blob_store as blob_store_module

        original_digest = blob_store_module.sha256_and_size

        def digest_with_event(path):
            events.append("digest")
            return original_digest(path)

        monkeypatch.setattr(blob_store_module, "sha256_and_size", digest_with_event)

        assert store.exists(key) is True
        assert events == ["snapshot", "digest"]

        manifest = BlobManifestV1.from_canonical_bytes(
            store.manifest_repository.get_raw(key) or b""
        )
        Path(manifest.locator).write_text("tamper! payload", encoding="utf-8")
        events.clear()

        with pytest.raises(CacheBlobPayloadTamperedError):
            store.exists(key)
        assert events == ["snapshot", "digest"]
        assert "handler" not in events
    finally:
        store.close()


def test_list_authenticates_every_selected_manifest_before_returning(tmp_path):
    """A conflicted selected record cannot be silently omitted from listing."""
    events: list[str] = []
    store = BlobStore(tmp_path / "list", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        first = store.put("first", key="selected-first", metadata={"group": "one"})
        second = store.put("second", key="selected-second", metadata={"group": "one"})
        _replace_signed_manifest(store, second, state="prepared")

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.list(prefix="selected-", metadata_filter={"group": "one"})

        assert events == []
        assert first == "selected-first"
    finally:
        store.close()


def test_update_metadata_resigns_only_user_metadata_and_rejects_structure(
    tmp_path, monkeypatch
):
    """Public metadata patches cannot alter signed storage structure."""
    store = BlobStore(tmp_path / "update", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler([]))

    try:
        key = store.put("update payload", key="update-key", metadata={"owner": "one"})
        before = BlobManifestV1.from_canonical_bytes(
            store.manifest_repository.get_raw(key) or b""
        )

        assert store.update_metadata(key, {"owner": "two", "label": "current"})

        after = BlobManifestV1.from_canonical_bytes(
            store.manifest_repository.get_raw(key) or b""
        )
        assert dict(after.user_metadata) == {"owner": "two", "label": "current"}
        assert after.signature != before.signature
        for field in (
            "schema_version",
            "key",
            "generation",
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

        raw_before_rejected_patch = store.manifest_repository.get_raw(key)
        assert raw_before_rejected_patch is not None
        monkeypatch.setattr(
            store.manifest_repository,
            "put_raw",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("structural patch must fail before manifest write")
            ),
        )

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.update_metadata(key, {"locator": "/unsafe-replacement"})
        assert store.manifest_repository.get_raw(key) == raw_before_rejected_patch
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
        original_publish = first_store.manifest_repository.publish_if_expected

        def block_first_publish(*args: Any, **kwargs: Any) -> None:
            publish_entered.set()
            assert release_first_publish.wait(timeout=2)
            original_publish(*args, **kwargs)

        monkeypatch.setattr(
            first_store.manifest_repository, "publish_if_expected", block_first_publish
        )

        def patch_first_store() -> None:
            try:
                first_result.append(first_store.update_metadata(key, {"owner": "first"}))
            except BaseException as exc:
                first_result.append(exc)

        first_thread = Thread(target=patch_first_store)
        first_thread.start()
        assert publish_entered.wait(timeout=2)

        _replace_signed_manifest(
            second_store,
            key,
            user_metadata={"owner": "second"},
        )
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


def test_update_metadata_rejects_an_authenticated_outside_root_locator(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Metadata mutation validates the signed locator before re-signing it."""
    store = BlobStore(tmp_path / "outside-locator", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler([]))
    try:
        key = store.put("payload", key="outside-key")
        _replace_signed_manifest(store, key, locator=str(tmp_path / "outside.bin"))
        raw_before = store.manifest_repository.get_raw(key)
        assert raw_before is not None
        monkeypatch.setattr(
            store.manifest_repository,
            "put_raw",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unsafe locator must fail before manifest mutation")
            ),
        )

        with pytest.raises(CacheUnsafePathError):
            store.update_metadata(key, {"label": "blocked"})

        assert store.manifest_repository.get_raw(key) == raw_before
    finally:
        store.close()


def test_delete_and_clear_preflight_authenticated_manifests_before_mutation(
    tmp_path, monkeypatch
):
    """Unsafe direct mutation never trusts a backend-shaped locator first."""
    store = BlobStore(tmp_path / "mutate", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler([]))

    try:
        delete_key = store.put("delete payload", key="delete-key")
        raw_manifest = store.manifest_repository.get_raw(delete_key)
        assert raw_manifest is not None
        payload_locator = Path(json.loads(raw_manifest)["locator"])
        tampered = json.loads(raw_manifest)
        tampered["signature"] = "0" * 64
        store.manifest_repository.put_raw(
            delete_key,
            json.dumps(
                tampered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
        )
        original_delete = store.guarded_handler_io.file_ops.delete

        def reject_only_tampered_payload(locator: Path | str) -> bool:
            if Path(locator) == payload_locator:
                raise AssertionError("unauthenticated manifest must not delete payload")
            return original_delete(locator)

        monkeypatch.setattr(
            store.guarded_handler_io.file_ops,
            "delete",
            reject_only_tampered_payload,
        )

        with pytest.raises(CacheBlobManifestUnauthenticatedError):
            store.delete(delete_key)
        assert store.manifest_repository.get_raw(delete_key) is not None

        first = store.put("first clear", key="clear-first")
        second = store.put("second clear", key="clear-second")
        _replace_signed_manifest(store, second, state="prepared")
        monkeypatch.setattr(
            store._clear_recovery,
            "clear",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("clear must not begin before full manifest preflight")
            ),
        )

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.clear()
        assert store.manifest_repository.get_raw(first) is not None
        assert store.manifest_repository.get_raw(second) is not None
        assert store.delete("absent-delete") is False
    finally:
        store.close()


@pytest.mark.parametrize(
    ("fault", "error_type"),
    (
        ("malformed", CacheBlobManifestMalformedError),
        ("future_schema", CacheBlobManifestUnsupportedVersionError),
        ("conflict", CacheBlobLifecycleConflictError),
    ),
)
def test_every_direct_read_surface_preserves_ordered_typed_failures(
    tmp_path, fault: str, error_type: type[Exception]
):
    """Present corrupt, future, and conflicted records never become misses."""
    store = BlobStore(tmp_path / fault, backend="json")
    try:
        key = store.put({"fault": fault}, key="fault-key")
        manifest = BlobManifestV1.from_canonical_bytes(
            store.manifest_repository.get_raw(key) or b""
        )
        payload_path = Path(manifest.locator)
        payload_before = payload_path.read_bytes()
        payload_mtime_before = payload_path.stat().st_mtime_ns
        if fault == "malformed":
            store.manifest_repository.put_raw(key, b"{")
        elif fault == "future_schema":
            future_manifest = json.loads(
                store.manifest_repository.get_raw(key) or b"{}"
            )
            future_manifest["schema_version"] = 2
            store.manifest_repository.put_raw(
                key,
                json.dumps(
                    future_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
        else:
            _replace_signed_manifest(store, key, state="prepared")
        raw_before = store.manifest_repository.get_raw(key)
        assert raw_before is not None

        for operation in (
            lambda: store.get(key),
            lambda: store.get_metadata(key),
            lambda: store.exists(key),
            lambda: store.list(),
        ):
            with pytest.raises(error_type):
                operation()
            assert store.manifest_repository.get_raw(key) == raw_before
            assert payload_path.read_bytes() == payload_before
            assert payload_path.stat().st_mtime_ns == payload_mtime_before
    finally:
        store.close()


def test_every_direct_read_surface_propagates_a_local_backend_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A backend fault remains typed across get, metadata, existence, and list."""
    store = BlobStore(tmp_path / "backend-failure", backend="json")
    try:
        key = store.put({"backend": "failure"}, key="backend-key")
        raw_before = store.manifest_repository.get_raw(key)
        assert raw_before is not None
        failure = CacheBlobBackendError("injected local repository failure")
        monkeypatch.setattr(
            store.manifest_repository,
            "get_raw",
            lambda _key: (_ for _ in ()).throw(failure),
        )
        monkeypatch.setattr(store.manifest_repository, "list_keys", lambda: [key])

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
def test_projection_backend_failures_are_typed_on_every_direct_surface(
    tmp_path, monkeypatch: pytest.MonkeyPatch, operation: str
):
    """Compatibility projection failures never bypass the BlobStore taxonomy."""
    store = BlobStore(tmp_path / operation, backend="json")
    try:
        key = store.put({"operation": operation}, key="projection-key")
        monkeypatch.setattr(store.manifest_repository, "list_keys", lambda: [key])
        failure = OSError("metadata projection unavailable")
        monkeypatch.setattr(
            store.manifest_repository,
            "list_backend_entries",
            lambda: (_ for _ in ()).throw(failure),
        )

        with pytest.raises(CacheBlobBackendError) as error:
            getattr(store, operation)()

        assert error.value.__cause__ is failure
    finally:
        store.close()
