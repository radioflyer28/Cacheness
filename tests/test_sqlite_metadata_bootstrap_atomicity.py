"""Atomic SQLite metadata-bootstrap contracts.

The worker schedules deliberately construct independent backends against a
genuinely absent file.  They use barriers rather than timing to make the
check-then-create window observable.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
import sqlite3
import threading

import pytest

from cacheness.error_handling import CacheIntegrityError, CacheLegacyFormatError
from cacheness.metadata import Base, SqliteBackend


_CURRENT_ENTRY_COLUMNS = (
    "cache_key",
    "description",
    "data_type",
    "prefix",
    "created_at",
    "accessed_at",
    "file_size",
    "file_hash",
    "entry_signature",
    "object_type",
    "storage_format",
    "serializer",
    "compression_codec",
    "actual_path",
    "cache_key_params",
)


def _assert_current_metadata_schema(database: Path) -> None:
    """Verify the durable metadata contract through an independent DBAPI read."""
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert tuple(
            row[1]
            for row in connection.execute("PRAGMA table_xinfo(cache_entries)")
        ) == _CURRENT_ENTRY_COLUMNS
        assert tuple(
            row[1]
            for row in connection.execute("PRAGMA table_xinfo(cache_stats)")
        ) == ("id", "cache_hits", "cache_misses", "last_updated")
        indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list(cache_entries)")
        }
        assert {
            "idx_list_entries",
            "idx_cleanup",
            "idx_size_mgmt",
            "idx_data_type",
        }.issubset(indexes)
        assert connection.execute("SELECT id FROM cache_stats").fetchall() == [(1,)]


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
    assert all(result == ("ok", None) for result in results[:-1]), results
    _assert_current_metadata_schema(database)


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
    _assert_current_metadata_schema(database)


def test_hostile_preexisting_schema_is_rejected_without_journal_mutation(
    tmp_path: Path,
) -> None:
    """Foreign evidence is never adopted as an empty metadata database."""
    database = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE foreign_records (id INTEGER PRIMARY KEY)")
        before_mode = connection.execute("PRAGMA journal_mode").fetchone()
        before_catalog = connection.execute(
            "SELECT type, name, tbl_name FROM sqlite_master ORDER BY name"
        ).fetchall()
    before_bytes = database.read_bytes()

    with pytest.raises(CacheIntegrityError) as error:
        SqliteBackend(str(database))

    assert error.value.context == {
        "reason": "metadata_corrupt",
        "backend": "sqlite",
        "operation": "metadata_bootstrap",
        "stage": "preflight",
        "path": str(database),
    }
    assert database.read_bytes() == before_bytes
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == before_mode
        assert connection.execute(
            "SELECT type, name, tbl_name FROM sqlite_master ORDER BY name"
        ).fetchall() == before_catalog


def test_unsupported_cache_entries_layout_retains_legacy_exception(tmp_path: Path) -> None:
    """The historical unsupported-layout public boundary is not relabeled."""
    database = tmp_path / "unsupported.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE cache_entries (wrong_column TEXT)")

    with pytest.raises(CacheLegacyFormatError) as error:
        SqliteBackend(str(database))

    assert error.value.context["reason"] == "unsupported_legacy_layout"


def test_post_ddl_failure_rolls_back_every_provisional_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DDL and validation share the explicit bootstrap transaction."""
    database = tmp_path / "rollback.sqlite3"

    def create_then_fail(connection, **_kwargs) -> None:
        connection.exec_driver_sql("CREATE TABLE bootstrap_sentinel (id INTEGER)")
        raise RuntimeError("forced bootstrap failure")

    monkeypatch.setattr(Base.metadata, "create_all", create_then_fail)
    with pytest.raises(RuntimeError, match="forced bootstrap failure"):
        SqliteBackend(str(database))

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'bootstrap_sentinel'"
        ).fetchone() is None
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name IN ('cache_entries', 'cache_stats')"
        ).fetchall() == []
