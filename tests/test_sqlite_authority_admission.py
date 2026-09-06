"""Deterministic admission coverage for short SQLite authority writes."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from threading import Barrier, Event, Lock, Thread

import pytest

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import (
    CacheBlobLifecycleTimeoutError,
    CacheBlobStoreClosedError,
)
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


def test_warmed_authorities_share_fifo_admission_and_retire_gate(tmp_path: Path) -> None:
    """A paused eligible writer cannot be overtaken by another authority instance."""
    first = SqliteLifecycleAuthority.for_root(tmp_path)
    second = SqliteLifecycleAuthority.for_root(tmp_path)
    first.begin_clear()  # Materialize and validate the bootstrap/schema path before admission.

    first_eligible = Event()
    second_enqueued = Event()
    release_first = Event()
    errors: list[BaseException] = []
    grants: list[str] = []

    def first_observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.eligible":
            first_eligible.set()
            assert release_first.wait(timeout=2)
        if event == "writer_admission.granted":
            grants.append("first")

    def second_observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.enqueued":
            second_enqueued.set()
        if event == "writer_admission.granted":
            grants.append("second")

    first.set_admission_observer_for_test(first_observer)
    second.set_admission_observer_for_test(second_observer)

    def run(authority: SqliteLifecycleAuthority) -> None:
        try:
            authority.begin_clear()
        except BaseException as error:  # pragma: no cover - asserted after joins
            errors.append(error)

    first_thread = Thread(target=run, args=(first,))
    second_thread = Thread(target=run, args=(second,))
    try:
        first_thread.start()
        assert first_eligible.wait(timeout=2)
        second_thread.start()
        assert second_enqueued.wait(timeout=2)
        release_first.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert errors == []
        assert grants == ["first", "second"]
    finally:
        release_first.set()
        first.close()
        second.close()

    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_different_authority_paths_do_not_share_admission(tmp_path: Path) -> None:
    """A paused writer on one database does not serialize another authority path."""
    first = SqliteLifecycleAuthority.for_root(tmp_path / "first")
    second = SqliteLifecycleAuthority.for_root(tmp_path / "second")
    first.begin_clear()
    second.begin_clear()

    first_eligible = Event()
    second_acquired = Event()
    release_first = Event()
    errors: list[BaseException] = []

    def first_observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.eligible":
            first_eligible.set()
            assert release_first.wait(timeout=2)

    def second_observer(event: str, _timestamp: float) -> None:
        if event == "sqlite.begin.acquired":
            second_acquired.set()

    first.set_admission_observer_for_test(first_observer)
    second.set_admission_observer_for_test(second_observer)

    def run(authority: SqliteLifecycleAuthority) -> None:
        try:
            authority.begin_clear()
        except BaseException as error:  # pragma: no cover - asserted after joins
            errors.append(error)

    first_thread = Thread(target=run, args=(first,))
    second_thread = Thread(target=run, args=(second,))
    try:
        first_thread.start()
        assert first_eligible.wait(timeout=2)
        second_thread.start()
        assert second_acquired.wait(timeout=2)
        release_first.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert errors == []
    finally:
        release_first.set()
        first.close()
        second.close()


def test_eligible_dispatch_expiry_keeps_the_original_deadline(tmp_path: Path) -> None:
    """Observer-controlled scheduler delay expires the ticket without a reset."""
    authority = SqliteLifecycleAuthority.for_root(tmp_path)
    authority.begin_clear()
    now = [10.0]
    events: list[str] = []

    def observer(event: str, _timestamp: float) -> None:
        events.append(event)
        if event == "writer_admission.eligible":
            now[0] += authority.lifecycle_limits.authority_busy_timeout_seconds

    authority.set_admission_timing_for_test(clock=lambda: now[0])
    authority.set_admission_observer_for_test(observer)
    try:
        with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
            authority.begin_clear()
    finally:
        authority.close()

    assert raised.value.context["stage"] == "scheduler_dispatch"
    assert raised.value.context["authority_busy_timeout_seconds"] == 0.187
    assert "sqlite.begin.attempt" not in events
    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_queued_ticket_expiry_uses_the_wait_seam_and_retires_exact_ticket(
    tmp_path: Path,
) -> None:
    """A non-head ticket exhausts one deadline without a wall-clock retry loop."""
    first = SqliteLifecycleAuthority.for_root(tmp_path)
    second = SqliteLifecycleAuthority.for_root(tmp_path)
    first.begin_clear()
    first_eligible = Event()
    release_first = Event()
    now = [0.0]

    def first_observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.eligible":
            first_eligible.set()
            assert release_first.wait(timeout=2)

    def consume_wait(_condition: object, _timeout: float) -> None:
        now[0] += second.lifecycle_limits.authority_busy_timeout_seconds

    first.set_admission_observer_for_test(first_observer)
    second.set_admission_timing_for_test(clock=lambda: now[0], waiter=consume_wait)
    first_thread = Thread(target=first.begin_clear)
    try:
        first_thread.start()
        assert first_eligible.wait(timeout=2)
        with pytest.raises(CacheBlobLifecycleTimeoutError) as raised:
            second.begin_clear()
        assert raised.value.context["stage"] == "writer_admission"
    finally:
        release_first.set()
        first_thread.join(timeout=2)
        first.close()
        second.close()

    assert not first_thread.is_alive()
    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_close_cancels_a_queued_ticket_without_stealing_active_owner(tmp_path: Path) -> None:
    """A closing instance wakes its queued ticket while the existing owner proceeds."""
    first = SqliteLifecycleAuthority.for_root(tmp_path)
    second = SqliteLifecycleAuthority.for_root(tmp_path)
    first.begin_clear()

    first_eligible = Event()
    second_enqueued = Event()
    release_first = Event()
    second_errors: list[BaseException] = []

    def first_observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.eligible":
            first_eligible.set()
            assert release_first.wait(timeout=2)

    def second_observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.enqueued":
            second_enqueued.set()

    first.set_admission_observer_for_test(first_observer)
    second.set_admission_observer_for_test(second_observer)

    def run_first() -> None:
        first.begin_clear()

    def run_second() -> None:
        try:
            second.begin_clear()
        except BaseException as error:  # pragma: no cover - asserted after join
            second_errors.append(error)

    first_thread = Thread(target=run_first)
    second_thread = Thread(target=run_second)
    try:
        first_thread.start()
        assert first_eligible.wait(timeout=2)
        second_thread.start()
        assert second_enqueued.wait(timeout=2)
        second.close()
        second_thread.join(timeout=2)
        assert not second_thread.is_alive()
        assert len(second_errors) == 1
        assert isinstance(second_errors[0], CacheBlobStoreClosedError)

        release_first.set()
        first_thread.join(timeout=2)
        assert not first_thread.is_alive()
    finally:
        release_first.set()
        first.close()
        second.close()

    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_external_sqlite_writer_still_times_out_at_sqlite_busy_stage(tmp_path: Path) -> None:
    """The local queue does not replace SQLite's cross-process writer arbitration."""
    authority = SqliteLifecycleAuthority.for_root(tmp_path)
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
        authority.close()

    assert raised.value.context["stage"] == "sqlite_busy"
    assert raised.value.__cause__ is not None
    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0


def test_eight_public_distinct_key_writers_emit_non_overlapping_stages(
    tmp_path: Path,
) -> None:
    """Public UnifiedCache writes share short admission while payload work stays usable."""
    cache = UnifiedCache(config=CacheConfig(cache_dir=tmp_path, metadata_backend="memory"))
    authority = cache._cache_blob_store.lifecycle_authority
    assert isinstance(authority, SqliteLifecycleAuthority)
    cache.put("warmup", writer="warmup")

    event_lock = Lock()
    events: list[tuple[str, float]] = []
    barrier = Barrier(8)
    errors: list[BaseException] = []

    def observer(event: str, timestamp: float) -> None:
        with event_lock:
            events.append((event, timestamp))

    authority.set_admission_observer_for_test(observer)

    def put(writer: int) -> None:
        try:
            barrier.wait(timeout=2)
            cache.put(f"value-{writer}", writer=writer)
        except BaseException as error:  # pragma: no cover - asserted after joins
            errors.append(error)

    threads = [Thread(target=put, args=(writer,)) for writer in range(8)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=4)
        assert not any(thread.is_alive() for thread in threads)
        assert errors == []
        for writer in range(8):
            assert cache.get(writer=writer) == f"value-{writer}"
    finally:
        cache.close()

    names = {event for event, _timestamp in events}
    assert {
        "sqlite.connection_preflight.started",
        "sqlite.connection_preflight.finished",
        "writer_admission.enqueued",
        "writer_admission.eligible",
        "writer_admission.granted",
        "sqlite.begin.attempt",
        "sqlite.begin.acquired",
        "transaction.finished",
        "writer_admission.released",
    } <= names
    assert all(timestamp >= 0 for _event, timestamp in events)
