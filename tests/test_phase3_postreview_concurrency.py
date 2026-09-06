"""Deterministic regressions for the final Phase 3 concurrency review."""

from __future__ import annotations

from threading import Event, Thread
from typing import Callable

from cacheness.config import LifecycleLimits
from cacheness.error_handling import CacheBlobCloseTimeoutError
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.coordination import InstanceAdmission


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

