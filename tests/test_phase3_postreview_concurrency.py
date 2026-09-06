"""Deterministic regressions for the final Phase 3 concurrency review."""

from __future__ import annotations

import ast
import json
from multiprocessing import get_context
import os
import sqlite3
from threading import Event, Thread
import time
from typing import Callable

import pytest

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobCloseTimeoutError,
    CacheBlobLifecycleTimeoutError,
)
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.coordination import InstanceAdmission
from cacheness.storage import sqlite_lifecycle_authority as sqlite_authority_module
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


class _AdvancingClock:
    """Controlled monotonic clock for admission-timeout schedules."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _join(thread: Thread) -> None:
    """Join a deterministic worker without allowing an unbounded test hang."""
    thread.join(timeout=5)
    assert not thread.is_alive()


def _warm_authority(authority: SqliteLifecycleAuthority) -> None:
    """Materialize an authority so fork schedules avoid bootstrap contention."""
    authority.begin_clear()


def _fork_child_authority_mutation(
    root: str,
    inherited: SqliteLifecycleAuthority,
    inherited_registry_lock: object,
    write_fd: int,
) -> None:
    """Report child-owned authority mutation results through one raw pipe."""
    fresh: SqliteLifecycleAuthority | None = None
    try:
        inherited_rejected = False
        try:
            inherited.begin_clear()
        except CacheBlobBackendError:
            inherited_rejected = True

        registry_before = SqliteLifecycleAuthority.admission_registry_size_for_test()
        lock_rebound = (
            sqlite_authority_module._WRITER_ADMISSION_REGISTRY_LOCK
            is not inherited_registry_lock
        )
        fresh = SqliteLifecycleAuthority.for_root(root)
        fresh.begin_clear()
        result: dict[str, object] = {
            "inherited_rejected": inherited_rejected,
            "lock_rebound": lock_rebound,
            "registry_before": registry_before,
            "registry_after": SqliteLifecycleAuthority.admission_registry_size_for_test(),
        }
    except BaseException as error:  # pragma: no cover - reported to parent assertions.
        result = {"error": f"{type(error).__name__}: {error}"}
    finally:
        if fresh is not None:
            fresh.close()
        os.write(write_fd, json.dumps(result).encode("utf-8"))
        os.close(write_fd)
        os._exit(0)


def _run_fork_child(
    root,
    inherited: SqliteLifecycleAuthority,
    inherited_registry_lock: object,
) -> dict[str, object]:
    """Run one fork child with a bounded join and return its pipe report."""
    read_fd, write_fd = os.pipe()
    child = get_context("fork").Process(
        target=_fork_child_authority_mutation,
        args=(str(root), inherited, inherited_registry_lock, write_fd),
    )
    child.start()
    os.close(write_fd)
    try:
        child.join(timeout=3)
        if child.is_alive():
            child.terminate()
            child.join(timeout=3)
        payload = os.read(read_fd, 4096)
    finally:
        os.close(read_fd)

    assert not child.is_alive()
    assert child.exitcode == 0
    return json.loads(payload)


def test_timed_out_clear_retires_its_exact_ticket_and_unblocks_fifo(
    tmp_path,
) -> None:
    """A queued public clear timeout cannot strand the next FIFO contender."""
    store = BlobStore(tmp_path / "timed-out-clear", backend="json")
    clock = _AdvancingClock()

    def advance_wait(condition, timeout: float) -> None:
        clock.value += timeout
        condition.wait(0)

    limits = LifecycleLimits(close_wait_seconds=0.187)
    admission = InstanceAdmission(
        limits,
        monotonic=clock,
        wait=advance_wait,
    )
    store._instance_admission = admission

    snapshot_entered = Event()
    release_snapshot = Event()
    first_result: list[int] = []
    first_errors: list[BaseException] = []
    second_errors: list[BaseException] = []
    third_result: list[int] = []
    third_errors: list[BaseException] = []

    def pause_first_snapshot(boundary: str) -> None:
        if boundary == "clear.snapshot_committed":
            snapshot_entered.set()
            assert release_snapshot.wait(timeout=5)

    def run_clear(result: list[int], errors: list[BaseException]) -> Callable[[], None]:
        def clear() -> None:
            try:
                result.append(store.clear())
            except BaseException as error:  # pragma: no cover - asserted below.
                errors.append(error)

        return clear

    store.lifecycle.test_hook = pause_first_snapshot
    first = Thread(target=run_clear(first_result, first_errors))
    first.start()
    assert snapshot_entered.wait(timeout=5)

    second = Thread(target=run_clear([], second_errors))
    second.start()
    _join(second)

    assert len(second_errors) == 1
    assert isinstance(second_errors[0], CacheBlobCloseTimeoutError)
    assert admission._active_clear is not None
    assert admission._snapshot_owner is admission._active_clear
    assert admission.in_flight == 1
    assert admission._clear_queue == []

    release_snapshot.set()
    _join(first)
    assert first_errors == []
    assert first_result == [0]

    third = Thread(target=run_clear(third_result, third_errors))
    third.start()
    _join(third)

    assert third_errors == []
    assert third_result == [0]
    assert admission._clear_queue == []
    assert admission._active_clear is None
    assert admission._snapshot_owner is None
    assert admission.in_flight == 0
    assert admission._admitted_threads == {}


def test_sqlite_begin_uses_only_the_remaining_absolute_busy_budget(tmp_path) -> None:
    """Observer dispatch plus SQLite contention cannot spend two deadline windows."""
    limits = LifecycleLimits(authority_busy_timeout_seconds=0.187)
    authority = SqliteLifecycleAuthority.for_root(
        tmp_path / "combined-deadline", lifecycle_limits=limits
    )
    _warm_authority(authority)
    blocker = sqlite3.connect(authority.path, isolation_level=None)
    observer_wait = Event()

    def observer(event: str, _timestamp: float) -> None:
        if event == "sqlite.begin.attempt":
            assert not observer_wait.wait(timeout=0.12)

    authority.set_admission_observer_for_test(observer)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
            authority.begin_clear()
        elapsed = time.monotonic() - started
    finally:
        if blocker.in_transaction:
            blocker.execute("ROLLBACK")
        blocker.close()
        authority.close()

    assert elapsed <= limits.authority_busy_timeout_seconds + 0.025
    assert raised.value.context["stage"] == "sqlite_busy"
    assert raised.value.context["authority_busy_timeout_seconds"] == 0.187
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_sqlite_busy_budget_milliseconds_never_round_up() -> None:
    """Sub-millisecond budget is immediate instead of extending the deadline."""
    assert SqliteLifecycleAuthority._busy_timeout_milliseconds(0.187) == 182
    assert SqliteLifecycleAuthority._busy_timeout_milliseconds(0.006999) == 1
    assert SqliteLifecycleAuthority._busy_timeout_milliseconds(0.005999) == 0
    assert SqliteLifecycleAuthority._busy_timeout_milliseconds(0.000999) == 0
    assert SqliteLifecycleAuthority._busy_timeout_milliseconds(0.0) == 0


def test_first_use_fifo_admits_before_connection_preflight_and_publishes_ready(
    tmp_path,
) -> None:
    """A first-use contender cannot open SQLite while the bootstrap owner holds EXCLUSIVE."""
    root = tmp_path / "first-use-admission"
    first = SqliteLifecycleAuthority.for_root(root)
    second = SqliteLifecycleAuthority.for_root(root)
    exclusive_acquired = Event()
    release_first = Event()
    second_enqueued = Event()
    second_preflight = Event()
    errors: list[BaseException] = []

    def bootstrap_boundary(boundary: str) -> None:
        if boundary == "authority.schema_initialize.exclusive_acquired":
            assert first.path.exists()
            assert first._schema_ready is False
            exclusive_acquired.set()
            assert release_first.wait(timeout=2)

    def second_observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.enqueued":
            second_enqueued.set()
        if event == "sqlite.connection_preflight.started":
            second_preflight.set()

    def run(authority: SqliteLifecycleAuthority) -> None:
        try:
            authority.begin_clear()
        except BaseException as error:  # pragma: no cover - asserted after joins.
            errors.append(error)

    first.set_bootstrap_hook_for_test(bootstrap_boundary)
    second.set_admission_observer_for_test(second_observer)
    first_thread = Thread(target=run, args=(first,))
    second_thread = Thread(target=run, args=(second,))
    try:
        first_thread.start()
        assert exclusive_acquired.wait(timeout=2)
        second_thread.start()
        assert second_enqueued.wait(timeout=2)
        assert not second_preflight.is_set()
        release_first.set()
        _join(first_thread)
        _join(second_thread)
    finally:
        release_first.set()
        first.close()
        second.close()

    assert errors == []
    assert first._schema_ready is True
    assert second._schema_ready is True
    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_preflight_busy_timeout_has_precise_stage_and_sqlite_cause(tmp_path) -> None:
    """External EXCLUSIVE contention cannot escape as lifecycle_authority_open."""
    root = tmp_path / "preflight-timeout"
    creator = SqliteLifecycleAuthority.for_root(root)
    creator.begin_clear()
    blocker = sqlite3.connect(creator.path, isolation_level=None)
    contender = SqliteLifecycleAuthority.for_root(root)
    try:
        blocker.execute("BEGIN EXCLUSIVE")
        with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
            contender.begin_clear()
    finally:
        if blocker.in_transaction:
            blocker.execute("ROLLBACK")
        blocker.close()
        creator.close()
        contender.close()

    assert raised.value.context["operation"] == "lifecycle_authority"
    assert raised.value.context["stage"] == "connection_configure"
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_forked_child_rebinds_registry_while_parent_gate_is_owned(tmp_path) -> None:
    """A fresh child cannot inherit the parent's granted same-path ticket."""
    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable on this platform")

    root = tmp_path / "forked-gate"
    warm = SqliteLifecycleAuthority.for_root(root)
    _warm_authority(warm)
    warm.close()
    parent = SqliteLifecycleAuthority.for_root(root)
    granted = Event()
    release_parent = Event()
    parent_errors: list[BaseException] = []

    def observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.granted":
            granted.set()
            assert release_parent.wait(timeout=5)

    def run_parent() -> None:
        try:
            parent.begin_clear()
        except BaseException as error:  # pragma: no cover - asserted below.
            parent_errors.append(error)

    parent.set_admission_observer_for_test(observer)
    parent_thread = Thread(target=run_parent)
    parent_thread.start()
    assert granted.wait(timeout=5)
    inherited_registry_lock = sqlite_authority_module._WRITER_ADMISSION_REGISTRY_LOCK
    try:
        child = _run_fork_child(root, parent, inherited_registry_lock)
    finally:
        release_parent.set()
        _join(parent_thread)
        parent.close()

    assert parent_errors == []
    assert child == {
        "inherited_rejected": True,
        "lock_rebound": True,
        "registry_before": 0,
        "registry_after": 0,
    }
    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_forked_child_never_acquires_an_inherited_registry_lock(tmp_path) -> None:
    """The child replaces a registry lock held by a parent helper at fork."""
    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable on this platform")

    root = tmp_path / "forked-registry-lock"
    warm = SqliteLifecycleAuthority.for_root(root)
    _warm_authority(warm)
    warm.close()
    inherited = SqliteLifecycleAuthority.for_root(root)
    lock_owned = Event()
    release_lock = Event()
    inherited_registry_lock = sqlite_authority_module._WRITER_ADMISSION_REGISTRY_LOCK

    def hold_registry_lock() -> None:
        with inherited_registry_lock:
            lock_owned.set()
            assert release_lock.wait(timeout=5)

    holder = Thread(target=hold_registry_lock)
    holder.start()
    assert lock_owned.wait(timeout=5)
    try:
        child = _run_fork_child(root, inherited, inherited_registry_lock)
    finally:
        release_lock.set()
        _join(holder)
        inherited.close()

    assert child == {
        "inherited_rejected": True,
        "lock_rebound": True,
        "registry_before": 0,
        "registry_after": 0,
    }
    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_authority_timeout_source_contract_has_one_canonical_context() -> None:
    """Every SQLite translation and backup boundary carries the caller's budget."""
    source = sqlite_authority_module.__file__
    assert source is not None
    tree = ast.parse(open(source, encoding="utf-8").read())
    translations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_translate_sqlite_error"
    ]
    assert translations
    required = {"operation", "stage", "deadline", "started_at"}
    for call in translations:
        keywords = {keyword.arg for keyword in call.keywords}
        assert required <= keywords
        operation = next(keyword.value for keyword in call.keywords if keyword.arg == "operation")
        assert isinstance(operation, ast.Constant)
        assert operation.value == "lifecycle_authority"

    backup_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backup"
    ]
    assert backup_calls
    assert all(
        any(
            keyword.arg == "sleep"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 0
            for keyword in call.keywords
        )
        for call in backup_calls
    )
    assert "lifecycle_authority_read" not in open(source, encoding="utf-8").read()
    assert "lifecycle_authority_expectation" not in open(source, encoding="utf-8").read()
    assert "lifecycle_projection_backup" not in open(source, encoding="utf-8").read()
    assert "lifecycle_authority_diagnostics" not in open(source, encoding="utf-8").read()


def test_public_read_entry_busy_has_canonical_stage_and_sqlite_cause(tmp_path) -> None:
    """A public authority read identifies its SQL boundary after preflight succeeds."""
    authority = SqliteLifecycleAuthority.for_root(tmp_path / "entry-read-boundary")
    _warm_authority(authority)
    blocker: sqlite3.Connection | None = None

    def block_entry_read(stage: str, _statement: str) -> None:
        nonlocal blocker
        if stage == "entry_read":
            blocker = sqlite3.connect(authority.path, isolation_level=None)
            blocker.execute("BEGIN EXCLUSIVE")

    authority.set_sql_boundary_observer_for_test(block_entry_read)
    try:
        with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
            authority.read_entry("missing")
    finally:
        if blocker is not None:
            if blocker.in_transaction:
                blocker.execute("ROLLBACK")
            blocker.close()
        authority.close()

    assert raised.value.context["operation"] == "lifecycle_authority"
    assert raised.value.context["stage"] == "entry_read"
    assert raised.value.context["authority_busy_timeout_seconds"] == 0.187
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
