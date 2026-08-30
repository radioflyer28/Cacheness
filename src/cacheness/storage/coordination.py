"""Small admission primitives for bounded BlobStore lifecycle transitions."""

from __future__ import annotations

from contextlib import contextmanager
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


__all__ = ["StoreAdmissionBarrier"]
