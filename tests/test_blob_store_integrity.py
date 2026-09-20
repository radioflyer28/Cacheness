"""Adversarial integrity contracts for canonical BlobStore records."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import logging
import json
import multiprocessing
import os
import inspect
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobManifestMalformedError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobPayloadMissingError,
    CacheBlobPayloadTamperedError,
    CacheBlobPayloadUnsupportedVersionError,
    CacheManifestIntegrityError,
    CacheReason,
    CacheUnsafePathError,
)
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.integrity import (
    ManifestKeyError,
    ManifestKeyDurabilityProvider,
    ManifestKeyProvider,
    sign_hmac_sha256,
    verify_hmac_sha256,
)
import cacheness.storage.integrity as integrity_module
from cacheness.storage.manifest import BlobManifest, sign_current_manifest
from cacheness.config import LifecycleLimits


def _store(
    root: Path, *, manifest_key_provider: object | None = None
) -> BlobStore:
    """Create the supported local topology while retaining key-provider injection."""
    return BlobStore(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        ),
        cache_dir=root,
        manifest_key_provider=manifest_key_provider,
    )


_KEY = b"0123456789abcdef0123456789abcdef"


def _leave_crash_partial_key(path: str) -> None:
    """Persist a short key in a child and terminate without cleanup."""
    key_path = Path(path)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, b"short")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os._exit(0)


def _hold_initialization_lock(
    lock_path: str,
    entered: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    """Hold a live POSIX initialization authority without touching key bytes."""
    if os.name == "nt":  # pragma: no cover - native Windows has its own adapter gate.
        os._exit(0)
    import fcntl

    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        entered.set()
        release.wait(timeout=10)
    finally:
        os.close(descriptor)


class _InjectedManifestKeyProvider:
    """Minimal application-owned provider for platform key-store integrations."""

    def get_key(self) -> bytes:
        return _KEY


class _RecordingKeyDurabilityProvider:
    """Application-owned first-key acknowledgement probe."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, tuple[int, int]]] = []

    def acknowledge_new_key(
        self, key_path: Path, expected_identity: tuple[int, int]
    ) -> None:
        self.calls.append((key_path, expected_identity))


def test_strict_key_provider_requires_explicit_initialization(tmp_path):
    """An absent file key cannot turn a read/reopen into a new trust root."""
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    provider = ManifestKeyProvider(key_path)

    with pytest.raises(ManifestKeyError) as error:
        provider.get_key()

    assert error.value.context["reason"] == CacheReason.MANIFEST_SIGNING_KEY_INVALID.value
    assert not key_path.exists()

    provider.initialize_new_store()
    assert provider.get_key() == key_path.read_bytes()
    assert len(key_path.read_bytes()) == 32


def test_key_provider_returns_the_attested_winner_of_a_first_write_race(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A normal exclusive-create race loads the winner instead of returning None."""
    import cacheness.storage.integrity as integrity_module

    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    provider = ManifestKeyProvider(key_path)
    original_open = integrity_module.os.open

    def winner_open(path, flags, *args):
        if flags & os.O_EXCL and Path(path) == key_path:
            Path(path).write_bytes(_KEY)
            Path(path).chmod(0o600)
            raise FileExistsError
        return original_open(path, flags, *args)

    monkeypatch.setattr(integrity_module.os, "open", winner_open)

    assert provider.initialize_new_store() == _KEY
    assert provider.get_key() == _KEY


def test_key_provider_removes_its_partial_key_when_persistence_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A failed first write leaves no malformed key to block later initialization."""
    import cacheness.storage.integrity as integrity_module

    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    provider = ManifestKeyProvider(key_path)
    monkeypatch.setattr(integrity_module.os, "write", lambda *_args: 0)

    with pytest.raises(ManifestKeyError):
        provider.initialize_new_store()

    assert not key_path.exists()


def test_key_provider_acknowledges_the_exact_new_trust_root_before_return(
    tmp_path: Path,
) -> None:
    """A file provider cannot return a new key before its owner acks it."""
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    durability = _RecordingKeyDurabilityProvider()
    provider = ManifestKeyProvider(key_path, durability_provider=durability)

    assert provider.initialize_new_store() == provider.get_key()
    assert len(durability.calls) == 1
    acknowledged_path, expected_identity = durability.calls[0]
    stat_result = key_path.stat()
    assert acknowledged_path == key_path
    assert expected_identity == (stat_result.st_dev, stat_result.st_ino)
    assert isinstance(durability, ManifestKeyDurabilityProvider)


def test_concurrent_first_key_call_waits_for_acknowledged_ready_record(tmp_path: Path) -> None:
    """No concurrent initializer receives a visible key before acknowledgement."""
    entered = threading.Event()
    release = threading.Event()
    results: list[bytes] = []

    class BlockingDurability:
        def acknowledge_new_key(self, _path: Path, _identity: tuple[int, int]) -> None:
            entered.set()
            assert release.wait(timeout=5)

    provider = ManifestKeyProvider(
        tmp_path / "blob_manifest_hmac_key.bin", durability_provider=BlockingDurability()
    )
    winner = threading.Thread(target=lambda: results.append(provider.get_or_initialize_new_store()))
    loser = threading.Thread(target=lambda: results.append(provider.get_or_initialize_new_store()))
    winner.start()
    assert entered.wait(timeout=5)
    loser.start()
    loser.join(timeout=0.05)
    assert loser.is_alive()
    assert results == []
    release.set()
    winner.join(timeout=5)
    loser.join(timeout=5)
    assert results == [provider.get_key(), provider.get_key()]


def test_same_process_initializer_honors_the_single_absolute_deadline(
    tmp_path: Path,
) -> None:
    """A local guard holder cannot mint a second initialization timeout budget."""
    import cacheness.storage.integrity as integrity_module

    entered = threading.Event()
    release = threading.Event()
    provider = ManifestKeyProvider(
        tmp_path / "blob_manifest_hmac_key.bin",
        durability_provider=type(
            "BlockingDurability",
            (),
            {
                "acknowledge_new_key": lambda _self, _path, _identity: (
                    entered.set(), release.wait(timeout=5)
                )[-1]
            },
        )(),
        lifecycle_limits=LifecycleLimits(
            key_initialization_timeout_seconds=0.05,
            key_initialization_retry_seconds=0.002,
        ),
    )
    winner = threading.Thread(target=provider.initialize_new_store)
    winner.start()
    assert entered.wait(timeout=5)
    started = time.monotonic()
    with pytest.raises(CacheBlobLifecycleTimeoutError):
        provider.initialize_new_store()
    assert time.monotonic() - started < 0.2
    assert not provider.key_path.with_suffix(".ready").exists()
    release.set()
    winner.join(timeout=5)
    assert not winner.is_alive()
    assert integrity_module._KeyInitializationGuardRegistry._entries == {}


def test_same_process_initializer_can_acquire_before_the_shared_deadline(
    tmp_path: Path,
) -> None:
    """Releasing local admission before expiry permits the same winner bytes."""
    import cacheness.storage.integrity as integrity_module

    entered = threading.Event()
    release = threading.Event()
    results: list[bytes] = []

    class BlockingDurability:
        def acknowledge_new_key(self, _path: Path, _identity: tuple[int, int]) -> None:
            entered.set()
            assert release.wait(timeout=5)

    provider = ManifestKeyProvider(
        tmp_path / "blob_manifest_hmac_key.bin",
        durability_provider=BlockingDurability(),
        lifecycle_limits=LifecycleLimits(
            key_initialization_timeout_seconds=1,
            key_initialization_retry_seconds=0.002,
        ),
    )
    winner = threading.Thread(target=lambda: results.append(provider.initialize_new_store()))
    contender = threading.Thread(target=lambda: results.append(provider.initialize_new_store()))
    winner.start()
    assert entered.wait(timeout=5)
    contender.start()
    release.set()
    winner.join(timeout=5)
    contender.join(timeout=5)
    assert not winner.is_alive()
    assert not contender.is_alive()
    assert results == [provider.get_key(), provider.get_key()]
    assert integrity_module._KeyInitializationGuardRegistry._entries == {}


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX child lock holder")
def test_live_key_initializer_times_out_without_reading_its_winner_bytes(
    tmp_path: Path,
) -> None:
    """A stalled external initializer ends at the policy deadline, not forever."""
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    lock_path = key_path.with_name(f"{key_path.name}.initializing.lock")
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    child = context.Process(
        target=_hold_initialization_lock,
        args=(str(lock_path), entered, release),
    )
    child.start()
    assert entered.wait(timeout=10)
    provider = ManifestKeyProvider(
        key_path,
        lifecycle_limits=LifecycleLimits(
            key_initialization_timeout_seconds=0.05,
            key_initialization_retry_seconds=0.005,
        ),
    )
    started = time.monotonic()
    try:
        with pytest.raises(CacheBlobLifecycleTimeoutError) as error:
            provider.initialize_new_store()
        assert time.monotonic() - started < 1
        assert error.value.context["reason"] == CacheReason.BLOB_LIFECYCLE_TIMEOUT.value
        assert not key_path.exists()
    finally:
        release.set()
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)
    assert child.exitcode == 0
    assert provider.initialize_new_store() == provider.get_key()


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX child lock holder")
def test_live_key_initializer_completes_when_the_holder_releases_before_deadline(
    tmp_path: Path,
) -> None:
    """A retrying contender acquires the same exact authority before timeout."""
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    lock_path = key_path.with_name(f"{key_path.name}.initializing.lock")
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    child = context.Process(
        target=_hold_initialization_lock,
        args=(str(lock_path), entered, release),
    )
    child.start()
    assert entered.wait(timeout=10)
    provider = ManifestKeyProvider(
        key_path,
        lifecycle_limits=LifecycleLimits(
            key_initialization_timeout_seconds=1,
            key_initialization_retry_seconds=0.005,
        ),
    )
    outcome: list[bytes] = []
    failure: list[BaseException] = []

    def initialize() -> None:
        try:
            outcome.append(provider.initialize_new_store())
        except BaseException as exc:  # pragma: no cover - parent assertion surfaces it.
            failure.append(exc)

    contender = threading.Thread(target=initialize)
    contender.start()
    time.sleep(0.02)
    release.set()
    contender.join(timeout=10)
    child.join(timeout=10)
    if child.is_alive():
        child.terminate()
        child.join(timeout=10)
    assert not contender.is_alive()
    assert child.exitcode == 0
    assert failure == []
    assert outcome == [provider.get_key()]


def test_initialization_guards_are_store_scoped_and_retire_after_parallel_use(
    tmp_path: Path,
) -> None:
    """Distinct keys reach acknowledgement concurrently and leave no guard residue."""
    import cacheness.storage.integrity as integrity_module

    barrier = threading.Barrier(2)
    entered = 0
    entered_guard = threading.Lock()

    class CoordinatedDurability:
        def acknowledge_new_key(self, _path: Path, _identity: tuple[int, int]) -> None:
            nonlocal entered
            with entered_guard:
                entered += 1
            barrier.wait(timeout=5)

    first = ManifestKeyProvider(
        tmp_path / "first" / "blob_manifest_hmac_key.bin",
        durability_provider=CoordinatedDurability(),
    )
    second = ManifestKeyProvider(
        tmp_path / "second" / "blob_manifest_hmac_key.bin",
        durability_provider=CoordinatedDurability(),
    )
    results: list[bytes] = []
    threads = [
        threading.Thread(target=lambda provider=provider: results.append(provider.initialize_new_store()))
        for provider in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert entered == 2
    assert len(results) == 2
    assert integrity_module._KeyInitializationGuardRegistry._entries == {}


def test_initialization_authority_uses_the_injected_nonblocking_win32_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Windows retry path requests FAIL_IMMEDIATELY before key inspection."""
    calls: list[tuple[bool, bool]] = []

    class FakeWindowsLockApi:
        def lock(
            self, _descriptor: int, *, exclusive: bool, nonblocking: bool = False
        ) -> object:
            calls.append((exclusive, nonblocking))
            return object()

        def unlock(self, _descriptor: int, _token: object) -> object:
            return None

    provider = ManifestKeyProvider(tmp_path / "blob_manifest_hmac_key.bin")
    provider.key_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(integrity_module, "_key_lock_platform", lambda: "nt")
    monkeypatch.setattr(integrity_module, "_windows_key_lock_api", FakeWindowsLockApi)

    with provider._initialization_lock():
        pass

    assert calls == [(True, True)]


def test_local_initialization_guard_never_retries_after_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock released by the scheduler after expiry cannot win a second probe."""
    import cacheness.storage.integrity as integrity_module

    clock = {"now": 0.0}

    class FakeTime:
        @staticmethod
        def monotonic() -> float:
            return clock["now"]

        @staticmethod
        def sleep(_duration: float) -> None:
            # Model a holder releasing exactly after the shared deadline.
            clock["now"] = 11.0
            lock.available = True

    class FakeLock:
        available = False
        attempts = 0

        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            self.attempts += 1
            return self.available

        def release(self) -> None:
            raise AssertionError("expired retry must not acquire the local guard")

    lock = FakeLock()
    identity = "deterministic-expiry"
    monkeypatch.setattr(integrity_module, "time", FakeTime)
    integrity_module._KeyInitializationGuardRegistry._entries[identity] = (lock, 0)
    try:
        with pytest.raises(CacheBlobLifecycleTimeoutError):
            with integrity_module._KeyInitializationGuardRegistry.acquire(
                identity, deadline=10.0
            ):
                raise AssertionError("expired retry must not enter the guard")
        assert lock.attempts == 1
    finally:
        integrity_module._KeyInitializationGuardRegistry._entries.pop(identity, None)


def test_windows_initialization_rejects_adapter_without_nonblocking_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old adapter cannot silently convert a deadline-bound retry to blocking."""
    class BlockingOnlyWindowsLockApi:
        def lock(self, _descriptor: int, *, exclusive: bool) -> object:
            return object()

        def unlock(self, _descriptor: int, _token: object) -> object:
            return None

    provider = ManifestKeyProvider(tmp_path / "blob_manifest_hmac_key.bin")
    provider.key_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(integrity_module, "_key_lock_platform", lambda: "nt")
    monkeypatch.setattr(
        integrity_module,
        "_windows_key_lock_api",
        BlockingOnlyWindowsLockApi,
    )

    with pytest.raises(CacheBlobBackendError) as error:
        with provider._initialization_lock(deadline=1.0):
            pass
    assert error.value.context["reason"] == CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED.value


def test_unacknowledged_key_is_resumed_after_provider_failure(tmp_path: Path) -> None:
    """A failed acknowledgement leaves an exact retryable inode, never authority."""
    calls = 0

    class FailOnceDurability:
        def acknowledge_new_key(self, _path: Path, _identity: tuple[int, int]) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("acknowledgement unavailable")

    provider = ManifestKeyProvider(
        tmp_path / "blob_manifest_hmac_key.bin", durability_provider=FailOnceDurability()
    )
    with pytest.raises(ManifestKeyError):
        provider.initialize_new_store()
    with pytest.raises(ManifestKeyError):
        provider.get_key()
    resumed = provider.get_or_initialize_new_store()
    assert resumed == provider.get_key()
    assert calls == 2


def test_unacknowledged_key_is_resumed_after_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A close failure leaves an exact but unusable key that a retry completes."""
    import cacheness.storage.integrity as integrity_module

    provider = ManifestKeyProvider(tmp_path / "blob_manifest_hmac_key.bin")
    original_close = integrity_module.os.close
    failed = False

    def fail_first_close(descriptor: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("close uncertain")
        original_close(descriptor)

    monkeypatch.setattr(integrity_module.os, "close", fail_first_close)
    with pytest.raises(ManifestKeyError):
        provider.initialize_new_store()
    monkeypatch.setattr(integrity_module.os, "close", original_close)
    assert provider.get_or_initialize_new_store() == provider.get_key()


def test_cross_process_crash_partial_key_is_retired_under_initialization_authority(
    tmp_path: Path,
) -> None:
    """A crash-short EEXIST key is repaired only after the shared lock is held."""
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=_leave_crash_partial_key, args=(str(key_path),))
    child.start()
    child.join(timeout=10)
    assert child.exitcode == 0

    provider = ManifestKeyProvider(key_path)
    assert provider.get_or_initialize_new_store() == provider.get_key()
    assert len(key_path.read_bytes()) == 32


def test_partial_ready_record_is_replaced_and_reacknowledged(tmp_path: Path) -> None:
    """Visible ready bytes are not completion until their durability step runs."""
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    provider = ManifestKeyProvider(key_path)
    key_path.write_bytes(_KEY)
    key_path.chmod(0o600)
    ready_path = key_path.with_name(f"{key_path.name}.ready")
    ready_path.write_bytes(b"partial")
    ready_path.chmod(0o600)

    assert provider.get_or_initialize_new_store() == _KEY
    assert provider.get_key() == _KEY
    assert ready_path.read_bytes() == provider._ready_bytes(_KEY, provider._read_existing_key_and_identity()[1])


def test_custom_manifest_key_provider_exception_is_translated(tmp_path: Path) -> None:
    """Every ordinary application-provider exception stays inside the public boundary."""
    class CustomProviderError(Exception):
        pass

    class ExplodingProvider:
        def get_key(self) -> bytes:
            raise CustomProviderError("keystore policy")

    store = _store(tmp_path, manifest_key_provider=ExplodingProvider())
    try:
        with pytest.raises(CacheBlobManifestUnauthenticatedError) as error:
            store.put({"value": "blocked"}, key="blocked")
        assert isinstance(error.value.__cause__, CustomProviderError)
        assert error.value.context["provider"] == "ExplodingProvider"
    finally:
        store.close()


def test_injected_key_provider_operational_failure_is_typed(tmp_path: Path) -> None:
    """Keystore outages cannot leak an untyped failure from a lifecycle put."""
    class UnavailableProvider:
        def get_key(self) -> bytes:
            raise OSError("keystore unavailable")

    store = _store(tmp_path, manifest_key_provider=UnavailableProvider())
    try:
        with pytest.raises(CacheBlobManifestUnauthenticatedError) as error:
            store.put({"value": "unavailable"}, key="unavailable")
        assert isinstance(error.value.__cause__, OSError)
        assert error.value.context["operation"] == "initialize_manifest_key"
    finally:
        store.close()


def test_noncanonical_signed_manifest_is_rejected_by_every_normal_boundary(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only JSON rewrites fail closed at every normal boundary."""
    store = signed_store
    key = "integrity-key"
    entry = store.lifecycle_authority.read_entry(key)
    assert entry is not None
    raw = entry.manifest
    noncanonical = json.dumps(json.loads(raw), indent=2).encode("utf-8")
    assert noncanonical != raw
    altered_entry = _authority_entry_with_raw_manifest(store, key, noncanonical)
    original_read_entry = _install_authority_entry(
        monkeypatch, store, key, altered_entry
    )

    operations = (
        lambda: store.get(key),
        lambda: store.put({"replacement": True}, key=key),
        lambda: store.delete(key),
        lambda: store.update_metadata(key, {"metadata": "rewrite"}),
        lambda: store.exists(key),
        lambda: store.list(),
    )
    for operation in operations:
        with pytest.raises(CacheBlobManifestMalformedError):
            operation()

    assert original_read_entry(key) == entry


def test_first_store_initialization_does_not_log_an_expected_missing_key(
    tmp_path, caplog: pytest.LogCaptureFixture
):
    """A successful first write does not emit a false public security alert."""
    caplog.set_level(logging.ERROR, logger="cacheness.error_handling")
    store = _store(tmp_path)
    try:
        store.put({"first": "write"}, key="first-key")
    finally:
        store.close()

    assert "Unable to read canonical manifest key" not in caplog.text


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode contract")
def test_strict_key_provider_rejects_symlink_and_unsafe_permissions(tmp_path):
    """A file-backed key is trusted only after no-follow regular-file attestation."""
    target = tmp_path / "target-key.bin"
    target.write_bytes(_KEY)
    target.chmod(0o600)
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    key_path.symlink_to(target)

    with pytest.raises(ManifestKeyError) as symlink_error:
        ManifestKeyProvider(key_path).get_key()

    assert symlink_error.value.context["reason"] == (
        CacheReason.MANIFEST_SIGNING_KEY_INVALID.value
    )

    key_path.unlink()
    key_path.write_bytes(_KEY)
    key_path.chmod(0o644)

    with pytest.raises(ManifestKeyError):
        ManifestKeyProvider(key_path).get_key()


def test_canonical_hmac_requires_exact_material_and_rejects_bad_signature(tmp_path):
    """Canonical signing uses exact supplied bytes and a fixed HMAC-SHA256 check."""
    provider = ManifestKeyProvider(tmp_path / "unused-key.bin", key=_KEY)
    payload = b'{"canonical":"manifest"}'
    signature = sign_hmac_sha256(payload, provider.get_key())

    assert verify_hmac_sha256(payload, signature, _KEY)
    assert not verify_hmac_sha256(payload, "0" * 64, _KEY)

    for invalid_key in (b"", b"too-short", b"x" * 33):
        with pytest.raises(ManifestKeyError) as error:
            ManifestKeyProvider(tmp_path / "invalid.bin", key=invalid_key)
        assert error.value.context["reason"] == (
            CacheReason.MANIFEST_SIGNING_KEY_INVALID.value
        )


def test_manifest_key_provider_has_no_non_posix_rejection_branch():
    """The one-user/session Windows topology can reach the file-key contract."""
    source = inspect.getsource(ManifestKeyProvider)

    assert 'os.name != "posix"' not in source
    assert "reparse" in source


def test_blob_store_accepts_an_injected_manifest_key_provider(tmp_path):
    """A platform key-store adapter can supply the narrow signing-key contract."""
    provider = _InjectedManifestKeyProvider()
    store = _store(tmp_path, manifest_key_provider=provider)
    try:
        assert store.put({"value": "injected"}, key="provider-key") == "provider-key"
        assert store.get("provider-key") == {"value": "injected"}
    finally:
        store.close()


def test_reopen_with_missing_key_never_creates_a_replacement_key(tmp_path):
    """A signed store with a missing key fails closed without evidence mutation."""
    store = _store(tmp_path)
    store.put({"value": "signed"}, key="signed")
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    authority_entry = store.lifecycle_authority.read_entry("signed")
    assert authority_entry is not None
    raw_manifest = authority_entry.manifest
    key_path.unlink()
    store.close()

    reopened = _store(tmp_path)
    try:
        with pytest.raises(CacheManifestIntegrityError) as error:
            reopened.get("signed")
    finally:
        reopened.close()

    assert error.value.context["reason"] == CacheReason.MANIFEST_SIGNATURE_INVALID.value
    assert not key_path.exists()
    final_store = _store(tmp_path)
    try:
        final_entry = final_store.lifecycle_authority.read_entry("signed")
        assert final_entry is not None
        assert raw_manifest == final_entry.manifest
    finally:
        final_store.close()


def _manifest_for(store: BlobStore, key: str) -> BlobManifest:
    """Load canonical manifest bytes from the committed lifecycle authority."""
    entry = store.lifecycle_authority.read_entry(key)
    assert entry is not None
    return BlobManifest.from_canonical_bytes(entry.manifest)


def _authority_entry_with_raw_manifest(
    store: BlobStore, key: str, raw_manifest: bytes, *, locator: str | None = None
) -> object:
    """Build a strict authority snapshot with controlled canonical raw bytes."""
    entry = store.lifecycle_authority.read_entry(key)
    assert entry is not None
    return replace(
        entry,
        manifest=raw_manifest,
        locator=entry.locator if locator is None else locator,
        expectation=replace(
            entry.expectation,
            manifest_digest=hashlib.sha256(raw_manifest).hexdigest(),
        ),
    )


def _install_authority_entry(
    monkeypatch: pytest.MonkeyPatch,
    store: BlobStore,
    key: str,
    altered_entry: object,
):
    """Expose one altered canonical entry through authority read and list seams."""
    original_read_entry = store.lifecycle_authority.read_entry
    original_list_entries = store.lifecycle_authority.list_entries
    monkeypatch.setattr(
        store.lifecycle_authority,
        "read_entry",
        lambda requested_key: (
            altered_entry
            if requested_key == key
            else original_read_entry(requested_key)
        ),
    )
    monkeypatch.setattr(
        store.lifecycle_authority,
        "list_entries",
        lambda: tuple(
            altered_entry if entry.key == key else entry
            for entry in original_list_entries()
        ),
    )
    return original_read_entry


def _replace_signed_manifest(
    store: BlobStore,
    lookup_key: str,
    **overrides: object,
) -> object:
    """Return an altered authenticated manifest through the authority seam."""
    current = _manifest_for(store, lookup_key)
    values = current.to_mapping(include_signature=False)
    values.update(overrides)
    altered = BlobManifest.from_mapping({**values, "signature": ""})
    signed = sign_current_manifest(altered, store._authority_manifest_key())
    locator = overrides.get("locator")
    return _authority_entry_with_raw_manifest(
        store,
        lookup_key,
        signed.canonical_bytes(),
        locator=locator if isinstance(locator, str) else None,
    )


@pytest.fixture
def signed_store(tmp_path):
    """Create a direct canonical record without retaining any test-global state."""
    store = _store(tmp_path)
    store.put({"value": "verified"}, key="integrity-key")
    try:
        yield store
    finally:
        store.close()


def test_read_authenticates_validates_snapshots_hashes_then_deserializes(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """M1 and M2 must authenticate and validate before payload use."""
    import cacheness.storage.blob_store as blob_store_module
    import cacheness.storage.lifecycle as lifecycle_module

    store = signed_store
    manifest = _manifest_for(store, "integrity-key")
    handler = store.handlers.get_handler_by_type(manifest.handler_type)
    events: list[str] = []

    original_verify = blob_store_module.verify_current_manifest
    original_resolve = store.handlers.resolve_payload_contract
    original_snapshot = store.guarded_handler_io.open_snapshot
    original_digest = lifecycle_module.sha256_and_size
    original_get = handler.get

    def verify_spy(*args, **kwargs):
        events.append("authenticate")
        return original_verify(*args, **kwargs)

    def resolve_spy(*args, **kwargs):
        events.append("validate")
        return original_resolve(*args, **kwargs)

    @contextmanager
    def snapshot_spy(*args, **kwargs):
        events.append("snapshot")
        with original_snapshot(*args, **kwargs) as snapshot:
            yield snapshot

    def digest_spy(*args, **kwargs):
        events.append("digest")
        return original_digest(*args, **kwargs)

    def get_spy(*args, **kwargs):
        events.append("handler")
        return original_get(*args, **kwargs)

    monkeypatch.setattr(blob_store_module, "verify_current_manifest", verify_spy)
    monkeypatch.setattr(store.handlers, "resolve_payload_contract", resolve_spy)
    monkeypatch.setattr(store.guarded_handler_io, "open_snapshot", snapshot_spy)
    monkeypatch.setattr(lifecycle_module, "sha256_and_size", digest_spy)
    monkeypatch.setattr(handler, "get", get_spy)

    assert store.get("integrity-key") == {"value": "verified"}
    assert events == [
        "authenticate",
        "validate",
        "snapshot",
        "authenticate",
        "digest",
        "handler",
    ]
    assert events.count("authenticate") == 2
    assert events.count("validate") == 1
    assert events.count("snapshot") == 1


@pytest.mark.parametrize("signature", ("", "0" * 64), ids=("absent", "wrong"))
def test_unauthenticated_manifest_fails_before_snapshot_or_handler(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
    signature: str,
):
    """An absent or wrong HMAC never authorizes locator or handler use."""
    store = signed_store
    manifest = _manifest_for(store, "integrity-key")
    if signature:
        tampered = manifest.with_signature(signature)
    else:
        values = manifest.to_mapping()
        values["signature"] = ""
        tampered = BlobManifest.from_mapping(values)
    original_entry = store.lifecycle_authority.read_entry("integrity-key")
    assert original_entry is not None
    tampered_entry = _authority_entry_with_raw_manifest(
        store, "integrity-key", tampered.canonical_bytes()
    )
    original_read_entry = _install_authority_entry(
        monkeypatch, store, "integrity-key", tampered_entry
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unauthenticated records must not snapshot or deserialize")

    monkeypatch.setattr(store.guarded_handler_io, "open_snapshot", forbidden)
    monkeypatch.setattr(store.handlers, "resolve_payload_contract", forbidden)

    with pytest.raises(CacheBlobManifestUnauthenticatedError) as error:
        store.get("integrity-key")

    assert error.value.context["reason"] == (
        CacheReason.BLOB_MANIFEST_UNAUTHENTICATED.value
    )
    assert original_read_entry("integrity-key") == original_entry


def test_unauthenticated_invalid_critical_syntax_is_not_semantically_classified(
    signed_store: BlobStore, monkeypatch: pytest.MonkeyPatch
):
    """A malformed-looking signed field cannot bypass the HMAC failure outcome."""
    store = signed_store
    record = _manifest_for(store, "integrity-key").to_mapping()
    record["digest"] = "not-a-sha256-digest"
    malformed_entry = _authority_entry_with_raw_manifest(
        store,
        "integrity-key",
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    _install_authority_entry(monkeypatch, store, "integrity-key", malformed_entry)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unauthenticated records must not use payload state")

    monkeypatch.setattr(store.guarded_handler_io, "open_snapshot", forbidden)
    monkeypatch.setattr(store.handlers, "resolve_payload_contract", forbidden)

    with pytest.raises(CacheBlobManifestMalformedError):
        store.get("integrity-key")


@pytest.mark.parametrize(
    ("overrides", "error_type"),
        (
            ({"key": "different-key"}, CacheBlobLifecycleConflictError),
            ({"state": "prepared"}, CacheBlobManifestMalformedError),
        ({"locator": "/outside-the-managed-root"}, CacheUnsafePathError),
    ),
    ids=("key", "lifecycle", "locator"),
)
def test_authenticated_critical_fields_fail_before_snapshot(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, str],
    error_type: type[Exception],
):
    """Authenticated key, lifecycle, and locator fields remain pre-snapshot gates."""
    store = signed_store
    if overrides.get("state") == "prepared":
        record = _manifest_for(store, "integrity-key").to_mapping()
        record.update(overrides)
        altered_entry = _authority_entry_with_raw_manifest(
            store,
            "integrity-key",
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
    else:
        altered_entry = _replace_signed_manifest(store, "integrity-key", **overrides)
    _install_authority_entry(monkeypatch, store, "integrity-key", altered_entry)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid critical fields must not snapshot payload")

    monkeypatch.setattr(store.guarded_handler_io, "open_snapshot", forbidden)

    with pytest.raises(error_type):
        store.get("integrity-key")


def test_unsupported_handler_contract_fails_before_snapshot(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """Authenticated handler identity is checked without opening payload bytes."""
    store = signed_store
    altered_entry = _replace_signed_manifest(
        store, "integrity-key", handler_type="unknown-handler"
    )
    _install_authority_entry(monkeypatch, store, "integrity-key", altered_entry)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsupported handler identity must not snapshot payload")

    monkeypatch.setattr(store.guarded_handler_io, "open_snapshot", forbidden)

    with pytest.raises(CacheBlobPayloadUnsupportedVersionError):
        store.get("integrity-key")


def test_missing_or_tampered_payload_is_typed_and_does_not_rewrite_evidence(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """Payload absence and tampering remain distinct from cache misses and cleanup."""
    store = signed_store
    manifest = _manifest_for(store, "integrity-key")
    payload_path = store.cache_dir / manifest.locator
    original_payload = payload_path.read_bytes()

    handler = store.handlers.get_handler_by_type(manifest.handler_type)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tampered payload must not reach the handler")

    monkeypatch.setattr(handler, "get", forbidden)

    payload_variants = (
        bytes([original_payload[0] ^ 1]) + original_payload[1:],
        original_payload[:-1],
        original_payload + b"x",
    )
    for modified_payload in payload_variants:
        payload_path.write_bytes(modified_payload)
        entry_before = store.lifecycle_authority.read_entry("integrity-key")
        assert entry_before is not None
        raw_before = entry_before.manifest
        key_before = (store.cache_dir / "blob_manifest_hmac_key.bin").read_bytes()
        mtime_before = payload_path.stat().st_mtime_ns

        with pytest.raises(CacheBlobPayloadTamperedError) as tampered_error:
            store.get("integrity-key")

        assert tampered_error.value.context["reason"] == (
            CacheReason.BLOB_PAYLOAD_TAMPERED.value
        )
        assert payload_path.read_bytes() == modified_payload
        assert payload_path.stat().st_mtime_ns == mtime_before
        current_entry = store.lifecycle_authority.read_entry("integrity-key")
        assert current_entry is not None
        assert current_entry.manifest == raw_before
        assert (store.cache_dir / "blob_manifest_hmac_key.bin").read_bytes() == key_before

    payload_path.unlink()
    with pytest.raises(CacheBlobPayloadMissingError) as missing_error:
        store.get("integrity-key")

    assert missing_error.value.context["reason"] == CacheReason.BLOB_PAYLOAD_MISSING.value
    assert not payload_path.exists()
    current_entry = store.lifecycle_authority.read_entry("integrity-key")
    assert current_entry is not None
    assert current_entry.manifest == raw_before
