"""Small admission primitives for bounded BlobStore lifecycle transitions."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Condition, Lock, get_ident
import time
from typing import Callable, Iterator

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobCloseTimeoutError,
    CacheBlobStoreClosedError,
)


class StoreAdmissionBarrier:
    """Coordinate only the finite clear-snapshot admission boundary.

    Ordinary operations share admission and remain concurrent with one another.
    A clear obtains aggregate admission long enough to persist its complete,
    authenticated target inventory, then releases it before reclaiming payloads.
    The barrier is process-local; manifest CAS remains the cross-process
    authority boundary.
    """

    _instances_guard = Lock()
    _instances: dict[str, "StoreAdmissionBarrier"] = {}

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._aggregate_active = False
        self._ordinary_active = 0

    @classmethod
    def for_root(cls, root: Path) -> "StoreAdmissionBarrier":
        """Return the one process-local barrier for an exact managed root."""
        identity = str(root.resolve())
        with cls._instances_guard:
            barrier = cls._instances.get(identity)
            if barrier is None:
                barrier = cls()
                cls._instances[identity] = barrier
            return barrier

    @contextmanager
    def ordinary_admission(self) -> Iterator[None]:
        """Admit one normal operation unless a snapshot is being established."""
        with self._condition:
            while self._aggregate_active:
                self._condition.wait()
            self._ordinary_active += 1
        try:
            yield
        finally:
            with self._condition:
                self._ordinary_active -= 1
                if self._ordinary_active == 0:
                    self._condition.notify_all()

    @contextmanager
    def aggregate_admission(self) -> Iterator[None]:
        """Exclude ordinary work only while creating a finite clear snapshot."""
        with self._condition:
            while self._aggregate_active:
                self._condition.wait()
            self._aggregate_active = True
            while self._ordinary_active:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._aggregate_active = False
                self._condition.notify_all()


@dataclass
class _KeyCoordinatorEntry:
    """One exact physical key's lock and in-flight acquisition count."""

    lock: Lock = field(default_factory=Lock)
    users: int = 0


class KeyCoordinatorRegistry:
    """Coordinate same-key local operations without serializing unrelated keys.

    The registry is deliberately owned by one ``BlobStore`` instance.  It
    supplies a cheap, deterministic in-process ordering boundary, while the
    manifest repository's exact compare-and-swap remains the authority for
    independent store instances and processes.  Entries are retained while a
    caller waits for the key lock, preventing a release/reacquire race from
    creating two locks for the same physical key.
    """

    def __init__(self) -> None:
        self._guard = Lock()
        self._entries: dict[str, _KeyCoordinatorEntry] = {}

    @property
    def size(self) -> int:
        """Return the number of currently acquired or awaited physical keys."""
        with self._guard:
            return len(self._entries)

    @contextmanager
    def hold(self, physical_key: str) -> Iterator[None]:
        """Acquire exactly one physical key and retire its entry after use."""
        with self._guard:
            entry = self._entries.get(physical_key)
            if entry is None:
                entry = _KeyCoordinatorEntry()
                self._entries[physical_key] = entry
            entry.users += 1

        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0 and self._entries.get(physical_key) is entry:
                    del self._entries[physical_key]

    @contextmanager
    def hold_many(self, physical_keys: Iterator[str]) -> Iterator[None]:
        """Acquire a set of keys in sorted order to avoid lock-order cycles."""
        with ExitStack() as stack:
            for physical_key in sorted(set(physical_keys)):
                stack.enter_context(self.hold(physical_key))
            yield


class InstanceState(str, Enum):
    """Lifecycle state for one BlobStore instance's owned resources."""

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class InstanceAdmission:
    """Admit instance work and serialize only the final close transition.

    This is deliberately separate from :class:`StoreAdmissionBarrier`: it
    protects an individual store's owned handles, while that barrier protects
    the short cross-instance clear snapshot boundary.  Normal work retains its
    existing per-key concurrency once admitted.
    """

    def __init__(
        self,
        lifecycle_limits: LifecycleLimits,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wait: Callable[[Condition, float], None] | None = None,
    ) -> None:
        self.lifecycle_limits = lifecycle_limits
        self._condition = Condition(Lock())
        self._state = InstanceState.OPEN
        self._in_flight = 0
        self._admitted_threads: dict[int, int] = {}
        self._release_in_progress = False
        self._monotonic = monotonic
        self._wait = self._condition_wait if wait is None else wait

    @staticmethod
    def _condition_wait(condition: Condition, timeout: float) -> None:
        """Wait through the condition seam so deadline tests need no sleep."""
        condition.wait(timeout)

    @property
    def state(self) -> InstanceState:
        """Return the current state without granting a new admission."""
        with self._condition:
            return self._state

    @property
    def in_flight(self) -> int:
        """Return admitted work count for deterministic lifecycle diagnostics."""
        with self._condition:
            return self._in_flight

    def require_open(self) -> None:
        """Reject a public entry point once the instance begins closing."""
        with self._condition:
            self._raise_if_not_open()

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Admit one operation and always release its drain reference."""
        thread_id = get_ident()
        with self._condition:
            self._raise_if_not_open()
            self._in_flight += 1
            self._admitted_threads[thread_id] = (
                self._admitted_threads.get(thread_id, 0) + 1
            )
        try:
            yield
        finally:
            with self._condition:
                self._in_flight -= 1
                remaining = self._admitted_threads[thread_id] - 1
                if remaining:
                    self._admitted_threads[thread_id] = remaining
                else:
                    del self._admitted_threads[thread_id]
                if self._in_flight == 0:
                    self._condition.notify_all()

    def begin_close(self) -> bool:
        """Start close and return whether this caller owns resource release.

        A timeout leaves the instance in ``CLOSING`` with resources live, so a
        later caller can drain and converge without reopening admission.
        """
        thread_id = get_ident()
        deadline = self._monotonic() + self.lifecycle_limits.close_wait_seconds
        with self._condition:
            if self._admitted_threads.get(thread_id, 0):
                raise CacheBlobCloseTimeoutError(
                    "BlobStore close cannot wait for work admitted by its own thread",
                    context={"state": self._state.value, "reentrant": True},
                )
            if self._state is InstanceState.CLOSED:
                return False
            if self._state is InstanceState.OPEN:
                self._state = InstanceState.CLOSING

            while self._in_flight or self._release_in_progress:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    raise CacheBlobCloseTimeoutError(
                        "BlobStore close timed out waiting for admitted work",
                        context={
                            "state": self._state.value,
                            "in_flight": self._in_flight,
                        },
                    )
                self._wait(self._condition, remaining)

            self._release_in_progress = True
            return True

    def finish_close(self, *, closed: bool) -> None:
        """Record a completed release attempt and wake concurrent close calls."""
        with self._condition:
            self._release_in_progress = False
            if closed:
                self._state = InstanceState.CLOSED
            self._condition.notify_all()

    def _raise_if_not_open(self) -> None:
        if self._state is not InstanceState.OPEN:
            raise CacheBlobStoreClosedError(
                "BlobStore is closing or closed",
                context={"state": self._state.value},
            )


__all__ = [
    "InstanceAdmission",
    "InstanceState",
    "KeyCoordinatorRegistry",
    "StoreAdmissionBarrier",
]
