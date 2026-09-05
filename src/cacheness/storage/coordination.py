"""Bounded process-local ordering and BlobStore instance admission.

LifecycleAuthority provides every cross-process correctness guarantee.  This
module only orders concurrent work inside one BlobStore instance and owns that
instance's admission/close state.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import Condition, Lock, get_ident
import time
from typing import Callable, Iterator

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobCloseTimeoutError,
    CacheBlobStoreClosedError,
)


@dataclass
class _KeyCoordinatorEntry:
    """One exact physical key's local mutex and acquisition count."""

    mutex: Lock = field(default_factory=Lock)
    users: int = 0


class KeyCoordinatorRegistry:
    """Order same-key operations within one instance without retaining keys.

    LifecycleAuthority, not this registry, arbitrates independent instances or
    processes. An entry remains present while a waiter is blocked so a later
    caller cannot receive a second mutex for the same key.
    """

    def __init__(self) -> None:
        self._guard = Lock()
        self._entries: dict[str, _KeyCoordinatorEntry] = {}

    @property
    def size(self) -> int:
        """Return the number of acquired or awaited keys."""
        with self._guard:
            return len(self._entries)

    @contextmanager
    def hold(self, physical_key: str) -> Iterator[None]:
        """Acquire one key and retire its entry when no caller uses it."""
        with self._guard:
            entry = self._entries.get(physical_key)
            if entry is None:
                entry = _KeyCoordinatorEntry()
                self._entries[physical_key] = entry
            entry.users += 1

        try:
            with entry.mutex:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0 and self._entries.get(physical_key) is entry:
                    del self._entries[physical_key]

    @contextmanager
    def hold_many(self, physical_keys: Iterator[str]) -> Iterator[None]:
        """Acquire keys in lexical order to avoid local ordering cycles."""
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
    """Admit instance work and serialize only its owned close transition."""

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
        """Wait through a seam so deadline tests need no sleep."""
        condition.wait(timeout)

    @property
    def state(self) -> InstanceState:
        """Return the current state without granting a new admission."""
        with self._condition:
            return self._state

    @property
    def in_flight(self) -> int:
        """Return admitted-work count for lifecycle diagnostics."""
        with self._condition:
            return self._in_flight

    def require_open(self) -> None:
        """Reject public work after this instance begins closing."""
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
        """Start close and say whether this caller owns resource release."""
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

            if self._state is InstanceState.CLOSED:
                return False

            self._release_in_progress = True
            return True

    def finish_close(self, *, closed: bool) -> None:
        """Record a release attempt and wake other close callers."""
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


__all__ = ["InstanceAdmission", "InstanceState", "KeyCoordinatorRegistry"]
