"""ADR-scoped concurrency regressions for the SQLite lifecycle authority."""

from __future__ import annotations

import json
from multiprocessing import get_context
import os
from pathlib import Path
import sqlite3
from threading import Event, Thread

import pytest

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleTimeoutError,
)
from cacheness.storage.sqlite_lifecycle_authority import (
    SQLITE_APPLICATION_ID,
    SqliteLifecycleAuthority,
)


def _join(thread: Thread) -> None:
    """Join a deterministic worker without allowing an unbounded test hang."""
    thread.join(timeout=5)
    assert not thread.is_alive()


def _warm_authority(authority: SqliteLifecycleAuthority) -> None:
    """Materialize the authority through one public lifecycle operation."""
    authority.begin_clear()


def _fork_ownership_child(
    root: str, inherited: SqliteLifecycleAuthority, fd: int
) -> None:
    """Prove inherited resources fail closed while a child-created authority works."""
    fresh: SqliteLifecycleAuthority | None = None
    try:
        try:
            inherited.begin_clear()
        except CacheBlobBackendError:
            inherited_rejected = True
        else:  # pragma: no cover - asserted by the parent process.
            inherited_rejected = False
        fresh = SqliteLifecycleAuthority.for_root(root)
        fresh.begin_clear()
        result: dict[str, object] = {
            "inherited_rejected": inherited_rejected,
            "fresh_succeeded": True,
        }
    except BaseException as error:  # pragma: no cover - reported to parent assertions.
        result = {"error": f"{type(error).__name__}: {error}"}
    finally:
        if fresh is not None:
            fresh.close()
        os.write(fd, json.dumps(result).encode("utf-8"))
        os.close(fd)
        os._exit(0)


def test_sqlite_contention_returns_a_contextual_retryable_timeout(
    tmp_path: Path,
) -> None:
    """Held SQLite authority contention is bounded without a FIFO success gate."""
    limits = LifecycleLimits(authority_busy_timeout_seconds=0.04)
    authority = SqliteLifecycleAuthority.for_root(
        tmp_path / "contended", lifecycle_limits=limits
    )
    _warm_authority(authority)
    blocker = sqlite3.connect(authority.path, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
            authority.begin_clear()
    finally:
        if blocker.in_transaction:
            blocker.execute("ROLLBACK")
        blocker.close()

    context = raised.value.context
    assert context["operation"] == "lifecycle_authority"
    assert context["authority_path"] == str(authority.path)
    assert (
        context["authority_busy_timeout_seconds"]
        == limits.authority_busy_timeout_seconds
    )
    assert context["retryable"] is True
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
    authority.begin_clear()
    authority.close()


def test_initialized_root_shared_workers_converge_through_sqlite(
    tmp_path: Path,
) -> None:
    """Independent workers progress after explicit SQLite root initialization."""
    root = tmp_path / "fresh-root"
    initializer = SqliteLifecycleAuthority.for_root(root)
    try:
        initializer.initialize()
        initializer_diagnostics = initializer.diagnostics()
        assert initializer_diagnostics["application_id"] == SQLITE_APPLICATION_ID
        initialized_store_identity = initializer_diagnostics["store_identity"]
        assert initialized_store_identity
    finally:
        initializer.close()

    first = SqliteLifecycleAuthority.for_root(root)
    second = SqliteLifecycleAuthority.for_root(root)
    start = Event()
    diagnostics: dict[str, dict[str, object]] = {}
    errors: dict[str, BaseException] = {}

    def begin_clear(name: str, authority: SqliteLifecycleAuthority) -> None:
        if not start.wait(timeout=5):
            errors[name] = AssertionError("initialized workers were not released")
            return
        try:
            authority.begin_clear()
            diagnostics[name] = authority.diagnostics()
        except BaseException as error:  # pragma: no cover - asserted below.
            errors[name] = error

    threads = [
        Thread(target=begin_clear, args=(name, authority))
        for name, authority in (("first", first), ("second", second))
    ]
    try:
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            _join(thread)

        assert errors == {}
        assert set(diagnostics) == {"first", "second"}
        assert all(
            worker_diagnostics["application_id"] == SQLITE_APPLICATION_ID
            for worker_diagnostics in diagnostics.values()
        )
        store_identities = {
            worker_diagnostics["store_identity"]
            for worker_diagnostics in diagnostics.values()
        }
        assert store_identities == {initialized_store_identity}
        first.begin_clear()
    finally:
        first.close()
        second.close()


def test_forked_authority_ownership_stays_fail_closed(tmp_path: Path) -> None:
    """Forked children construct their own authority instead of reusing a parent instance."""
    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable on this platform")

    root = tmp_path / "fork-ownership"
    inherited = SqliteLifecycleAuthority.for_root(root)
    _warm_authority(inherited)
    read_fd, write_fd = os.pipe()
    child = get_context("fork").Process(
        target=_fork_ownership_child,
        args=(str(root), inherited, write_fd),
    )
    child.start()
    os.close(write_fd)
    try:
        child.join(timeout=5)
        payload = os.read(read_fd, 4096)
    finally:
        os.close(read_fd)
        inherited.close()

    assert not child.is_alive()
    assert child.exitcode == 0
    assert json.loads(payload) == {"inherited_rejected": True, "fresh_succeeded": True}
