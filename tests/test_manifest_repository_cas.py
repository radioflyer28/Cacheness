"""Exact-record conditional publication contracts for local manifests."""

from __future__ import annotations

import builtins
import multiprocessing
import os
from pathlib import Path
from threading import Barrier, Lock, Thread

import pytest

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheUnsafePathError,
)
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend
from cacheness.storage.manifest_repository import (
    InMemoryManifestRepository,
    JsonManifestRepository,
    ManifestCursor,
    ManifestExpectation,
    SqliteManifestRepository,
)
from cacheness.storage import coordination
from cacheness.storage import path_security
from cacheness.storage import manifest_repository as manifest_repository_module
from cacheness.storage.path_security import ManagedFileOps, resolve_managed_locator


def _record(label: str) -> bytes:
    """Return deliberately opaque canonical-record stand-ins for repository tests."""
    return f"canonical-manifest-record::{label}".encode("utf-8")


def _race_json_manifest_cas_process(
    metadata_path: str,
    record: bytes,
    ready: multiprocessing.queues.Queue,
    release: multiprocessing.synchronize.Event,
    outcomes: multiprocessing.queues.Queue,
) -> None:
    """Race a fresh JSON authority repository without shared Python locks."""
    backend = JsonBackend(metadata_path)
    repository = JsonManifestRepository(backend)
    try:
        ready.put("ready")
        if not release.wait(timeout=10):
            outcomes.put("timeout")
            return
        repository.publish_if_expected("key", ManifestExpectation.absent(), record)
        outcomes.put("won")
    except CacheBlobLifecycleConflictError:
        outcomes.put("conflict")
    except BaseException as exc:  # pragma: no cover - surfaced by parent assertion.
        outcomes.put(f"error:{exc!r}")
    finally:
        repository.close()
        backend.close()


def _publish_json_while_parent_swaps_lock_name(
    metadata_path: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
    outcomes: multiprocessing.queues.Queue,
) -> None:
    """Hold the original descriptor while a parent swaps only its lock pathname."""
    backend = JsonBackend(metadata_path)
    repository = JsonManifestRepository(backend)
    repository.after_json_lock_acquisition = lambda: (ready.set(), release.wait(timeout=10))
    try:
        repository.publish_if_expected(
            "key", ManifestExpectation.absent(), _record("original-authority")
        )
        outcomes.put("original-won")
    except BaseException as exc:  # pragma: no cover - surfaced by parent assertion.
        outcomes.put(f"original-error:{exc!r}")
    finally:
        repository.close()
        backend.close()


def _attempt_json_publish_after_lock_swap(
    metadata_path: str, outcomes: multiprocessing.queues.Queue
) -> None:
    """A later opener must reject the replacement lock before it can enter CAS."""
    backend = JsonBackend(metadata_path)
    try:
        repository = JsonManifestRepository(backend)
    except CacheUnsafePathError:
        outcomes.put("replacement-blocked")
    except BaseException as exc:  # pragma: no cover - surfaced by parent assertion.
        outcomes.put(f"replacement-error:{exc!r}")
    else:
        try:
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record("replacement-authority")
            )
            outcomes.put("replacement-won")
        except CacheBlobLifecycleConflictError:
            outcomes.put("replacement-conflict")
        finally:
            repository.close()
    finally:
        backend.close()


def _repository_pair(tmp_path: Path, backend_name: str):
    """Build independently constructed repositories over one local topology."""
    if backend_name == "memory":
        backend = InMemoryBackend()
        return (
            InMemoryManifestRepository(backend),
            InMemoryManifestRepository(backend),
            (backend,),
        )
    if backend_name == "json":
        metadata_path = tmp_path / "metadata.json"
        first_backend = JsonBackend(metadata_path)
        second_backend = JsonBackend(metadata_path)
        return (
            JsonManifestRepository(first_backend),
            JsonManifestRepository(second_backend),
            (first_backend, second_backend),
        )
    metadata_path = tmp_path / "metadata.db"
    first_backend = SqliteBackend(metadata_path)
    second_backend = SqliteBackend(metadata_path)
    return (
        SqliteManifestRepository(first_backend),
        SqliteManifestRepository(second_backend),
        (first_backend, second_backend),
    )


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_create_if_absent_has_one_independent_repository_winner(
    tmp_path: Path, backend_name: str
) -> None:
    """A second absence contender loses without altering the exact winner bytes."""
    first_repo, second_repo, backends = _repository_pair(tmp_path, backend_name)
    winner = _record("winner")
    loser = _record("loser")
    try:
        first_repo.publish_if_expected(
            "key", ManifestExpectation.absent(), winner
        )

        with pytest.raises(CacheBlobLifecycleConflictError):
            second_repo.publish_if_expected(
                "key", ManifestExpectation.absent(), loser
            )

        assert first_repo.get_raw("key") == winner
        assert second_repo.get_raw("key") == winner
    finally:
        for backend in backends:
            backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_exact_record_cas_rejects_same_generation_stale_patch(
    tmp_path: Path, backend_name: str
) -> None:
    """A record digest prevents last-write-wins updates within one generation."""
    first_repo, second_repo, backends = _repository_pair(tmp_path, backend_name)
    original = _record("generation-one-metadata-one")
    winner = _record("generation-one-metadata-two")
    stale_loser = _record("generation-one-metadata-three")
    expectation = ManifestExpectation.from_authenticated_record("generation-one", original)
    try:
        first_repo.put_raw("key", original)
        first_repo.publish_if_expected("key", expectation, winner)

        with pytest.raises(CacheBlobLifecycleConflictError):
            second_repo.publish_if_expected("key", expectation, stale_loser)

        assert first_repo.get_raw("key") == winner
        assert second_repo.get_raw("key") == winner
    finally:
        for backend in backends:
            backend.close()


def test_json_cas_has_one_exact_cross_process_winner(tmp_path: Path) -> None:
    """The JSON authority boundary is a real interprocess winner selection."""
    metadata_path = tmp_path / "concurrent.json"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    release = context.Event()
    outcomes = context.Queue()
    records = (_record("process-a"), _record("process-b"))
    workers = [
        context.Process(
            target=_race_json_manifest_cas_process,
            args=(str(metadata_path), record, ready, release, outcomes),
        )
        for record in records
    ]
    for worker in workers:
        worker.start()
    try:
        assert ready.get(timeout=10) == "ready"
        assert ready.get(timeout=10) == "ready"
        release.set()
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert sorted(outcomes.get(timeout=10) for _ in workers) == ["conflict", "won"]

        backend = JsonBackend(metadata_path)
        repository = JsonManifestRepository(backend)
        try:
            assert repository.get_raw("key") in records
        finally:
            repository.close()
            backend.close()
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def test_json_lock_name_swap_cannot_create_a_second_cross_process_authority(
    tmp_path: Path,
) -> None:
    """A replacement opener is rejected while the original descriptor publishes."""
    root = tmp_path / "json-lock-swap-process"
    root.mkdir()
    metadata_path = root / "metadata.json"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    outcomes = context.Queue()
    original = context.Process(
        target=_publish_json_while_parent_swaps_lock_name,
        args=(str(metadata_path), ready, release, outcomes),
    )
    original.start()
    assert ready.wait(timeout=10)

    lock_path = root / ".metadata.json.manifest-cas.lock"
    replacement = root / ".replacement-lock"
    replacement.write_bytes(b"lock\n")
    os.replace(replacement, lock_path)
    contender = context.Process(
        target=_attempt_json_publish_after_lock_swap,
        args=(str(metadata_path), outcomes),
    )
    contender.start()
    contender.join(timeout=10)
    assert contender.exitcode == 0

    release.set()
    original.join(timeout=10)
    assert original.exitcode == 0
    assert sorted(outcomes.get(timeout=5) for _ in range(2)) == [
        "original-won",
        "replacement-blocked",
    ]

    # The old descriptor was permitted to finish; the intentionally replaced
    # name is then a fail-closed topology, never a new authority to adopt.
    backend = JsonBackend(metadata_path)
    try:
        with pytest.raises(CacheUnsafePathError):
            JsonManifestRepository(backend)
    finally:
        backend.close()


def test_json_cas_uses_the_win32_adapter_when_fcntl_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulated Windows path never imports POSIX locking and has one winner."""
    shared_lock = Lock()
    calls: list[str] = []

    class FakeWindowsLockApi:
        def lock(self, _descriptor: int, *, exclusive: bool) -> object:
            assert exclusive
            shared_lock.acquire()
            calls.append("lock")
            return object()

        def unlock(self, _descriptor: int, _token: object) -> object:
            calls.append("unlock")
            shared_lock.release()
            return None

    original_import = builtins.__import__

    def reject_fcntl(name: str, *args: object, **kwargs: object):
        if name == "fcntl":
            raise ImportError("fcntl is unavailable on Windows")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(coordination, "_platform_name", lambda: "nt")
    monkeypatch.setattr(coordination, "_windows_lock_api", FakeWindowsLockApi)
    monkeypatch.setattr(builtins, "__import__", reject_fcntl)

    metadata_path = tmp_path / "windows.json"
    first_backend = JsonBackend(metadata_path)
    second_backend = JsonBackend(metadata_path)
    first = JsonManifestRepository(first_backend)
    second = JsonManifestRepository(second_backend)
    gate = Barrier(2)
    outcomes: list[str] = []

    def contender(repository: JsonManifestRepository, record: bytes) -> None:
        gate.wait(timeout=5)
        try:
            repository.publish_if_expected("key", ManifestExpectation.absent(), record)
            outcomes.append("won")
        except CacheBlobLifecycleConflictError:
            outcomes.append("conflict")

    threads = [
        Thread(target=contender, args=(first, _record("windows-a"))),
        Thread(target=contender, args=(second, _record("windows-b"))),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert sorted(outcomes) == ["conflict", "won"]
        assert calls.count("lock") >= 2
        assert calls.count("unlock") == calls.count("lock")
    finally:
        first.close()
        second.close()
        first_backend.close()
        second_backend.close()


@pytest.mark.parametrize("substitute_during_open", (False, True))
def test_json_authority_lock_rejects_symlink_substitution_without_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitute_during_open: bool,
) -> None:
    """A lock symlink cannot win before or between managed create/open checks."""
    outside = tmp_path / "outside-lock"
    root = tmp_path / "inside"
    root.mkdir()
    backend = JsonBackend(root / "metadata.json")
    file_ops = ManagedFileOps(root)
    lock_locator = resolve_managed_locator(
        root,
        ".metadata.json.manifest-cas.lock",
        operation="manifest_repository_authority_lock",
        allow_missing_leaf=True,
    )
    try:
        if substitute_during_open:
            def swap_lock(operation: str, locator: Path) -> None:
                if operation == "read_stream" and locator == lock_locator:
                    lock_locator.unlink()
                    lock_locator.symlink_to(outside)

            monkeypatch.setattr(file_ops, "before_operation", swap_lock)
        else:
            lock_locator.symlink_to(outside)

        with pytest.raises(CacheUnsafePathError):
            JsonManifestRepository(backend, file_ops=file_ops)
        assert not outside.exists()
        assert backend._metadata.get("entries", {}).get("key") is None
    finally:
        file_ops.close()
        backend.close()


def test_json_authority_lock_rejects_a_live_lock_name_replacement(
    tmp_path: Path,
) -> None:
    """A retained lock descriptor rejects later name/inode substitution."""
    root = tmp_path / "live-lock-substitution"
    root.mkdir()
    outside = tmp_path / "outside-lock"
    backend = JsonBackend(root / "metadata.json")
    repository = JsonManifestRepository(backend)
    assert repository._json_lock_locator is not None
    lock_locator = repository._json_lock_locator
    try:
        lock_locator.unlink()
        lock_locator.symlink_to(outside)
        with pytest.raises(CacheBlobBackendError):
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record("blocked")
            )
        assert not outside.exists()
        assert backend._metadata.get("entries", {}).get("key") is None
    finally:
        repository.close()
        backend.close()


@pytest.mark.parametrize("seam", ("validation", "acquisition"))
def test_json_authority_never_publishes_into_a_replaced_root(
    tmp_path: Path, seam: str
) -> None:
    """Descriptor-anchored JSON CAS rejects either root-replacement boundary."""
    root = tmp_path / f"replaced-json-root-{seam}"
    root.mkdir()
    retired = tmp_path / f"retired-json-root-{seam}"
    backend = JsonBackend(root / "metadata.json")
    repository = JsonManifestRepository(backend)

    def replace_root() -> None:
        root.rename(retired)
        root.mkdir()

    if seam == "validation":
        repository.after_json_lock_validation = replace_root
    else:
        repository.after_json_lock_acquisition = replace_root
    try:
        with pytest.raises(CacheBlobBackendError):
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record(f"blocked-{seam}")
            )
        assert not (root / "metadata.json").exists()
        assert not (retired / "metadata.json").exists()
    finally:
        repository.close()
        backend.close()


def test_json_authority_lock_rejects_a_hard_link_from_outside_the_store(
    tmp_path: Path,
) -> None:
    """An outside hard link cannot become the canonical JSON authority inode."""
    root = tmp_path / "hard-linked-json-lock"
    root.mkdir()
    outside = tmp_path / "outside-lock"
    outside.write_bytes(b"lock\n")
    backend = JsonBackend(root / "metadata.json")
    lock_locator = root / ".metadata.json.manifest-cas.lock"
    os.link(outside, lock_locator)
    try:
        with pytest.raises(CacheUnsafePathError):
            JsonManifestRepository(backend)
        assert outside.stat().st_nlink == 2
        assert not (root / "metadata.json").exists()
    finally:
        backend.close()


def test_json_authority_lock_rejects_regular_inode_replacement(
    tmp_path: Path,
) -> None:
    """A regular-file swap cannot partition a retained authority descriptor."""
    root = tmp_path / "regular-json-lock-swap"
    root.mkdir()
    backend = JsonBackend(root / "metadata.json")
    repository = JsonManifestRepository(backend)
    assert repository._json_lock_locator is not None
    replacement = tmp_path / "replacement-lock"
    replacement.write_bytes(b"replacement\n")
    try:
        os.replace(replacement, repository._json_lock_locator)
        with pytest.raises(CacheBlobBackendError):
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record("regular-swap")
            )
        assert not (root / "metadata.json").exists()
    finally:
        repository.close()
        backend.close()


def test_json_repository_uses_native_windows_control_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulated Windows path uses only the injectable native directory API."""
    root = tmp_path / "windows-json"
    root.mkdir()
    backend = JsonBackend(root / "metadata.json")
    calls: list[str] = []

    class FakeWindowsFileApi:
        def rename_no_replace(self, temporary: Path, destination: Path) -> None:
            calls.append("rename")
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(temporary, destination)

        def flush_directory(self, directory: Path) -> None:
            assert directory == root or directory == root / ".cacheness-lock-authorities"
            calls.append("flush_directory")

    monkeypatch.setattr(path_security, "_platform_name", lambda: "nt")
    monkeypatch.setattr(path_security, "_windows_file_api", FakeWindowsFileApi)
    try:
        repository = JsonManifestRepository(backend)
        try:
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record("windows-control")
            )
            assert repository.get_raw("key") == _record("windows-control")
            assert "flush_directory" in calls
        finally:
            repository.close()
    finally:
        backend.close()


@pytest.mark.parametrize("failure", ("create", "open", "identity"))
def test_failed_direct_json_repository_construction_closes_its_owned_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Every direct-construction failure releases the repository-owned descriptor."""
    root = tmp_path / f"owned-root-close-{failure}"
    root.mkdir()
    backend = JsonBackend(root / "metadata.json")
    file_ops = ManagedFileOps(root)
    closed: list[bool] = []
    original_close = file_ops.close

    def track_close() -> None:
        closed.append(True)
        original_close()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"injected {failure} failure")

    monkeypatch.setattr(file_ops, "close", track_close)
    monkeypatch.setattr(manifest_repository_module, "ManagedFileOps", lambda _root: file_ops)
    if failure == "create":
        monkeypatch.setattr(file_ops, "ensure_lifecycle_lock", fail)
    elif failure == "open":
        monkeypatch.setattr(file_ops, "open_verified_regular_file", fail)
    else:
        monkeypatch.setattr(file_ops, "ensure_lifecycle_lock", fail)
    try:
        with pytest.raises(OSError, match=failure):
            JsonManifestRepository(backend)
        assert closed == [True]
    finally:
        backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_remove_if_expected_cannot_retire_a_replaced_record(
    tmp_path: Path, backend_name: str
) -> None:
    """Stale reclamation preserves a record published after the observed bytes."""
    first_repo, second_repo, backends = _repository_pair(tmp_path, backend_name)
    original = _record("old")
    replacement = _record("new")
    old_expectation = ManifestExpectation.from_authenticated_record("old", original)
    new_expectation = ManifestExpectation.from_authenticated_record("new", replacement)
    try:
        first_repo.put_raw("key", original)
        first_repo.publish_if_expected("key", old_expectation, replacement)

        with pytest.raises(CacheBlobLifecycleConflictError):
            second_repo.remove_if_expected("key", old_expectation)

        assert first_repo.get_raw("key") == replacement
        second_repo.remove_if_expected("key", new_expectation)
        assert first_repo.get_raw("key") is None
    finally:
        for backend in backends:
            backend.close()


def test_sqlite_cas_rolls_back_compatibility_projection_with_raw_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw-row failure leaves both SQLite projections at their prior winner."""
    backend = SqliteBackend(tmp_path / "metadata.db")
    repository = SqliteManifestRepository(backend)
    original = _record("original")
    replacement = _record("replacement")
    original_entry = {"description": "original", "data_type": "object"}
    replacement_entry = {"description": "replacement", "data_type": "object"}
    repository.put_raw("key", original, entry_data=original_entry)
    expectation = ManifestExpectation.from_authenticated_record("generation", original)

    def fail_raw_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected raw record failure")

    monkeypatch.setattr(repository, "_write_raw_row", fail_raw_write)
    try:
        with pytest.raises(CacheBlobBackendError):
            repository.publish_if_expected(
                "key", expectation, replacement, entry_data=replacement_entry
            )

        assert repository.get_raw("key") == original
        assert backend.get_entry("key")["description"] == "original"
    finally:
        backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_manifest_pages_are_stable_bounded_and_retain_the_supplied_limits(
    tmp_path: Path, backend_name: str
) -> None:
    """Local adapters expose two-item opaque pages without a local policy copy."""
    repository, _other, backends = _repository_pair(tmp_path, backend_name)
    limits = LifecycleLimits(manifest_page_size=2)
    repository = type(repository)(repository.backend, lifecycle_limits=limits)
    records = {key: _record(key) for key in ("delta", "alpha", "charlie", "bravo")}
    try:
        for key, record in records.items():
            repository.put_raw(key, record)

        first = repository.list_page()
        assert repository.lifecycle_limits is limits
        assert [key for key, _raw in first.entries] == ["alpha", "bravo"]
        assert [raw for _key, raw in first.entries] == [records["alpha"], records["bravo"]]
        assert first.next_cursor == ManifestCursor("bravo")

        second = repository.list_page(first.next_cursor)
        assert [key for key, _raw in second.entries] == ["charlie", "delta"]
        assert second.next_cursor is None
    finally:
        for backend in backends:
            backend.close()
