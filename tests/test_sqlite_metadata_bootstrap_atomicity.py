"""Atomic SQLite metadata-bootstrap contracts.

The worker schedules deliberately construct independent backends against a
genuinely absent file.  They use barriers rather than timing to make the
check-then-create window observable.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
import threading

from cacheness.metadata import SqliteBackend


def _construct_backend(
    path: str,
    barrier,
    outcomes,
) -> None:
    """Construct one independent backend and report its bounded outcome."""
    backend = None
    try:
        barrier.wait(timeout=20)
        backend = SqliteBackend(path)
        outcomes.put(("ok", None))
    except BaseException as error:  # pragma: no cover - parent asserts outcome.
        outcomes.put(("error", f"{type(error).__name__}: {error}"))
    finally:
        if backend is not None:
            backend.close()


def _thread_race_child(path: str, outcomes) -> None:
    """Run the thread schedule in a disposable spawned interpreter."""
    workers = 64
    barrier = threading.Barrier(workers)
    threads = [
        threading.Thread(
            target=_construct_backend,
            args=(path, barrier, outcomes),
            daemon=True,
        )
        for _ in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    outcomes.put(("threads_finished", all(not thread.is_alive() for thread in threads)))


def test_threaded_fresh_metadata_constructors_converge(tmp_path: Path) -> None:
    """64 independent constructors must not expose SQLAlchemy's DDL race."""
    database = tmp_path / "threaded.sqlite3"
    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    child = context.Process(target=_thread_race_child, args=(str(database), outcomes))
    child.start()
    child.join(timeout=60)
    assert child.exitcode == 0

    results = [outcomes.get(timeout=5) for _ in range(65)]
    assert results[-1] == ("threads_finished", True)
    assert all(result == ("ok", None) for result in results[:-1])


def test_spawned_fresh_metadata_constructors_converge(tmp_path: Path) -> None:
    """Separate spawned processes must converge without process-local locking."""
    database = tmp_path / "processes.sqlite3"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(8)
    outcomes = context.Queue()
    workers = [
        context.Process(target=_construct_backend, args=(str(database), barrier, outcomes))
        for _ in range(8)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    assert [outcomes.get(timeout=5) for _ in workers] == [("ok", None)] * len(workers)
