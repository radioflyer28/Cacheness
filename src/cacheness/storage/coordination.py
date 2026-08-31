"""Small admission primitives for bounded BlobStore lifecycle transitions."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Lock
from typing import Iterator


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


__all__ = ["KeyCoordinatorRegistry", "StoreAdmissionBarrier"]
