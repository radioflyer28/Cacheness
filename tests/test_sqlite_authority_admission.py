"""Progress and ownership coverage for SQLite lifecycle authority contention."""

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
    CacheBlobStoreClosedError,
)
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


def _join(thread: Thread) -> None:
    """Join one test worker without permitting a silent hang."""
    thread.join(timeout=5)
    assert not thread.is_alive()


def _fork_ownership_child(root: str, inherited: SqliteLifecycleAuthority, fd: int) -> None:
    """Report inherited-resource rejection and fresh-child authority success."""
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
    except Exception as error:  # pragma: no cover - reported to the parent process.
        result = {"error": f"{type(error).__name__}: {error}"}
    finally:
        if fresh is not None:
            fresh.close()
        os.write(fd, json.dumps(result).encode("utf-8"))
        os.close(fd)
        os._exit(0)


def test_held_sqlite_writer_times_out_then_later_retry_succeeds(tmp_path: Path) -> None:
    """SQLite contention is bounded and retryable without a same-process queue."""
    limits = LifecycleLimits(authority_busy_timeout_seconds=0.04)
    authority = SqliteLifecycleAuthority.for_root(tmp_path, lifecycle_limits=limits)
    authority.begin_clear()
    blocker = sqlite3.connect(authority.path, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
            authority.begin_clear()
    finally:
        if blocker.in_transaction:
            blocker.execute("ROLLBACK")
        blocker.close()

    assert raised.value.context["retryable"] is True
    assert raised.value.context["authority_path"] == str(authority.path)
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
    authority.begin_clear()
    authority.close()


def test_initialized_authorities_converge_without_admission_registry(
    tmp_path: Path,
) -> None:
    """Initialized authorities coordinate through SQLite, not FIFO tickets."""
    first = SqliteLifecycleAuthority.for_root(tmp_path / "fresh")
    second = SqliteLifecycleAuthority.for_root(tmp_path / "fresh")
    # Bootstrap is an explicit deployment boundary. Concurrent first creation
    # is intentionally outside the supported progress guarantee in ADR 0001.
    first.initialize()
    start = Event()
    errors: list[Exception] = []

    def initialize(authority: SqliteLifecycleAuthority) -> None:
        start.wait(timeout=5)
        try:
            authority.begin_clear()
        except Exception as error:  # pragma: no cover - asserted after joins.
            errors.append(error)

    threads = [Thread(target=initialize, args=(authority,)) for authority in (first, second)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        _join(thread)
    try:
        assert errors == []
        assert first.diagnostics()["store_identity"] == second.diagnostics()["store_identity"]
    finally:
        first.close()
        second.close()


def test_close_rejects_new_work_without_reclaiming_authority_state(tmp_path: Path) -> None:
    """Close is instance ownership only and leaves committed authority intact."""
    authority = SqliteLifecycleAuthority.for_root(tmp_path)
    authority.begin_clear()
    path = authority.path
    authority.close()

    with pytest.raises(CacheBlobStoreClosedError):
        authority.begin_clear()

    reopened = SqliteLifecycleAuthority.for_root(tmp_path)
    try:
        assert reopened.diagnostics()["store_identity"]
        assert path.exists()
    finally:
        reopened.close()


def test_forked_children_create_fresh_authorities(tmp_path: Path) -> None:
    """A child process never reuses the parent-owned SQLite authority instance."""
    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable on this platform")

    root = tmp_path / "fork"
    inherited = SqliteLifecycleAuthority.for_root(root)
    inherited.begin_clear()
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
