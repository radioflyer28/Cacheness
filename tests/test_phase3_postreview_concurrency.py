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
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


def _join(thread: Thread) -> None:
    """Join a deterministic worker without allowing an unbounded test hang."""
    thread.join(timeout=5)
    assert not thread.is_alive()


def _warm_authority(authority: SqliteLifecycleAuthority) -> None:
    """Materialize the authority through one public lifecycle operation."""
    authority.begin_clear()


def _fork_ownership_child(root: str, inherited: SqliteLifecycleAuthority, fd: int) -> None:
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


def test_sqlite_contention_returns_a_contextual_retryable_timeout(tmp_path: Path) -> None:
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
    assert context["authority_busy_timeout_seconds"] == limits.authority_busy_timeout_seconds
    assert context["retryable"] is True
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
    authority.begin_clear()
    authority.close()


def test_fresh_root_bootstrap_converges_through_sqlite(tmp_path: Path) -> None:
    """Independent first users converge without a process-global admission queue."""
    root = tmp_path / "fresh-root"
    first = SqliteLifecycleAuthority.for_root(root)
    second = SqliteLifecycleAuthority.for_root(root)
    start = Event()
    errors: list[BaseException] = []

    def initialize(authority: SqliteLifecycleAuthority) -> None:
        start.wait(timeout=5)
        try:
            authority.begin_clear()
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    threads = [
        Thread(target=initialize, args=(authority,)) for authority in (first, second)
    ]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        _join(thread)
    try:
        assert errors == []
        assert first.diagnostics()["application_id"] == second.diagnostics()["application_id"]
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
