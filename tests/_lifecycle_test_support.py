"""Deterministic support for Phase 3 lifecycle authority tests.

This module deliberately contains only test seams.  It snapshots local evidence
without creating directories or SQLite sidecars, and it records lifecycle
boundaries without using sleeps as an ordering oracle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
from threading import Event


class InjectedLifecycleFault(RuntimeError):
    """A deterministic fault raised from an armed lifecycle boundary."""


class BoundaryHooks:
    """Observe or fail named authority and payload boundaries deterministically."""

    def __init__(self) -> None:
        self._observers: list[Callable[[str], None]] = []
        self._faults: dict[str, BaseException] = {}
        self.reached: list[str] = []

    def add_observer(self, observer: Callable[[str], None]) -> None:
        """Record an observer that runs synchronously at every boundary."""
        self._observers.append(observer)

    def arm_fault(self, boundary: str, fault: BaseException | None = None) -> None:
        """Make one future reach of ``boundary`` raise a deterministic fault."""
        self._faults[boundary] = fault or InjectedLifecycleFault(boundary)

    def reach(self, boundary: str) -> None:
        """Notify observers, then raise the boundary's one-shot injected fault."""
        self.reached.append(boundary)
        for observer in self._observers:
            observer(boundary)
        fault = self._faults.pop(boundary, None)
        if fault is not None:
            raise fault


class ReleaseGate:
    """A test-only event gate used to prove ordering without timing assumptions."""

    def __init__(self) -> None:
        self.arrived = Event()
        self.released = Event()

    def wait_at_boundary(self, timeout: float = 5.0) -> None:
        """Signal arrival and wait for an explicit test release."""
        self.arrived.set()
        if not self.released.wait(timeout):
            raise TimeoutError("test did not release the lifecycle boundary")

    def release(self) -> None:
        """Release a worker paused at its deterministic boundary."""
        self.released.set()


def run_python_subprocess(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run one isolated Python lifecycle probe with captured deterministic output."""
    return subprocess.run(
        [sys.executable, *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


@dataclass(frozen=True)
class AuthorityPathState:
    """One bounded, non-following filesystem observation for an authority root."""

    relative_path: str
    object_type: str
    mode: int
    mtime_ns: int
    size: int
    sha256: str | None


def _object_type(file_stat: os.stat_result) -> str:
    if stat.S_ISDIR(file_stat.st_mode):
        return "directory"
    if stat.S_ISREG(file_stat.st_mode):
        return "regular"
    if stat.S_ISLNK(file_stat.st_mode):
        return "symlink"
    return "other"


def _regular_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        while chunk := os.read(descriptor, 64 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def authority_root_snapshot(root: Path) -> tuple[AuthorityPathState, ...]:
    """Return a deterministic bounded snapshot without creating or following paths."""
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return ()

    paths = [root]
    if stat.S_ISDIR(root_stat.st_mode):
        paths.extend(sorted(root.rglob("*")))

    states: list[AuthorityPathState] = []
    for path in paths:
        file_stat = path.lstat()
        object_type = _object_type(file_stat)
        states.append(
            AuthorityPathState(
                relative_path="." if path == root else path.relative_to(root).as_posix(),
                object_type=object_type,
                mode=stat.S_IMODE(file_stat.st_mode),
                mtime_ns=file_stat.st_mtime_ns,
                size=file_stat.st_size,
                sha256=_regular_sha256(path) if object_type == "regular" else None,
            )
        )
    return tuple(states)


def classify_authority_evidence(root: Path) -> str:
    """Classify root evidence without creating an authority or opening SQLite.

    This Wave 0 classifier is intentionally a test contract for future authority
    construction.  It makes all artifact classes explicit before Plan 02 wires
    the runtime's zero-mutation inspection policy.
    """
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        return "empty"
    if not stat.S_ISDIR(root_stat.st_mode):
        return "wrong_root_object"

    entries = {path.name: path for path in root.iterdir()}
    if not entries:
        return "empty"
    if "lifecycle-authority-v2.sqlite3" in entries:
        return "future_authority"
    if "lifecycle-authority-v1.sqlite3" in entries:
        return "corrupt_authority"
    if "provenance.json" in entries:
        return "legacy_without_authority"
    if any(name.startswith(".cacheness-inventory-") for name in entries):
        return "scheduler_without_authority"
    has_payload = any(name.endswith((".bin", ".pkl", ".npz", ".parquet")) for name in entries)
    has_metadata = any(
        name in {"cache_metadata.json", "cache_metadata.db"} for name in entries
    )
    if has_payload and has_metadata:
        return "mixed_without_authority"
    if has_payload:
        return "payload_without_authority"
    if has_metadata:
        return "metadata_without_authority"
    return "wrong_root_object"
