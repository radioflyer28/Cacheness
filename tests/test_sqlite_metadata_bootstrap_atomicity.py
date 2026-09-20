"""Explicit SQLite lifecycle-authority initialization contracts.

The legacy metadata adapter is intentionally gone. These stable-path checks
cover the current BlobStore/SqliteLifecycleAuthority initialization boundary
without treating concurrent first-use as an availability guarantee.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
import sqlite3
import threading

import pytest

from cacheness.error_handling import CacheBlobMigrationRequiredError
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.sqlite_lifecycle_authority import (
    AUTHORITY_RELATIVE_PATH,
    SQLITE_APPLICATION_ID,
    SQLITE_USER_VERSION,
    SqliteLifecycleAuthority,
)


_CURRENT_TABLES = {
    "authority_state",
    "cleanup_debt",
    "clear_runs",
    "clear_targets",
    "entries",
    "entry_lineage",
    "mutations",
    "reconciliation_actions",
    "reconciliation_runs",
    "store_identity",
}


def _local_topology(
    root: Path, authority: SqliteLifecycleAuthority | None = None
) -> StoreTopology:
    """Build exactly the supported local payload/authority pairing."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=(
            BackendRef(instance=authority)
            if authority is not None
            else BackendRef(name="sqlite", options={"root": root})
        ),
    )


def _assert_current_authority_schema(root: Path) -> None:
    """Inspect the durable authority independently of its implementation object."""
    database = root / AUTHORITY_RELATIVE_PATH
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert connection.execute("PRAGMA application_id").fetchone() == (
            SQLITE_APPLICATION_ID,
        )
        assert connection.execute("PRAGMA user_version").fetchone() == (
            SQLITE_USER_VERSION,
        )
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert _CURRENT_TABLES.issubset(tables)
        assert len(connection.execute("SELECT identity FROM store_identity").fetchall()) == 1
        revisions = connection.execute("SELECT revision FROM authority_state").fetchall()
        assert len(revisions) == 1
        assert isinstance(revisions[0][0], int)


def _initialize_existing_authority(root: str, barrier, outcomes) -> None:
    """Use independent explicit authority instances after offline initialization."""
    authority = SqliteLifecycleAuthority.for_root(root)
    try:
        barrier.wait(timeout=20)
        authority.initialize()
        outcomes.put(("ok", None))
    except BaseException as error:  # pragma: no cover - parent asserts outcome.
        outcomes.put(("error", f"{type(error).__name__}: {error}"))
    finally:
        authority.close()


def _thread_race_child(root: str, outcomes) -> None:
    """Run independent initialized-root authorities in a fresh interpreter."""
    workers = 16
    barrier = threading.Barrier(workers)
    threads = [
        threading.Thread(
            target=_initialize_existing_authority,
            args=(root, barrier, outcomes),
            daemon=True,
        )
        for _ in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    outcomes.put(("threads_finished", all(not thread.is_alive() for thread in threads)))


def test_explicit_blob_store_initialization_creates_current_authority_schema(
    tmp_path: Path,
) -> None:
    """Schema creation occurs at explicit initialization before shared workers."""
    root = tmp_path / "initialized"
    with BlobStore(_local_topology(root), cache_dir=root) as store:
        store.initialize()
        assert store.put({"value": "round-trip"}, key="entry") == "entry"
        assert store.get("entry") == {"value": "round-trip"}

    _assert_current_authority_schema(root)


def test_threaded_initialized_authorities_converge_without_first_use_claims(
    tmp_path: Path,
) -> None:
    """Independent authorities reopen an explicitly initialized root safely."""
    root = tmp_path / "threaded.sqlite"
    with BlobStore(_local_topology(root), cache_dir=root) as initializer:
        initializer.initialize()

    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    child = context.Process(target=_thread_race_child, args=(str(root), outcomes))
    child.start()
    child.join(timeout=60)
    assert child.exitcode == 0

    results = [outcomes.get(timeout=5) for _ in range(17)]
    assert results[-1] == ("threads_finished", True)
    assert all(result == ("ok", None) for result in results[:-1]), results
    _assert_current_authority_schema(root)


def test_spawned_initialized_authorities_converge_without_process_local_state(
    tmp_path: Path,
) -> None:
    """Separate processes reopen one prepared authority without a global lock."""
    root = tmp_path / "processes.sqlite"
    with BlobStore(_local_topology(root), cache_dir=root) as initializer:
        initializer.initialize()

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    outcomes = context.Queue()
    workers = [
        context.Process(
            target=_initialize_existing_authority,
            args=(str(root), barrier, outcomes),
        )
        for _ in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    assert [outcomes.get(timeout=5) for _ in workers] == [("ok", None)] * len(workers)
    _assert_current_authority_schema(root)


def test_foreign_root_is_rejected_without_mutating_evidence(tmp_path: Path) -> None:
    """An established non-authority root is migration evidence, never bootstrap input."""
    root = tmp_path / "foreign"
    root.mkdir()
    foreign = root / "foreign-records.sqlite3"
    foreign.write_bytes(b"foreign evidence")
    before = foreign.read_bytes()

    authority = SqliteLifecycleAuthority.for_root(root)
    try:
        with pytest.raises(CacheBlobMigrationRequiredError):
            authority.initialize()
    finally:
        authority.close()

    assert foreign.read_bytes() == before
    assert not (root / AUTHORITY_RELATIVE_PATH).exists()


def test_incomplete_authority_layout_is_rejected_without_implicit_upgrade(
    tmp_path: Path,
) -> None:
    """An obsolete authority leaf remains unchanged until offline migration/rebuild."""
    root = tmp_path / "obsolete"
    database = root / AUTHORITY_RELATIVE_PATH
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE legacy_entries (id INTEGER PRIMARY KEY)")
    before = database.read_bytes()

    authority = SqliteLifecycleAuthority.for_root(root)
    try:
        with pytest.raises(CacheBlobMigrationRequiredError):
            authority.initialize()
    finally:
        authority.close()

    assert database.read_bytes() == before


def test_pre_commit_schema_failure_rolls_back_provisional_tables(
    tmp_path: Path,
) -> None:
    """Schema initialization uses one SQLite transaction and leaves no partial tables."""
    root = tmp_path / "rollback"
    authority = SqliteLifecycleAuthority.for_root(root)

    def fail_after_exclusive_lock(boundary: str) -> None:
        if boundary == "authority.schema_initialize.exclusive_acquired":
            raise RuntimeError("forced bootstrap failure")

    authority.set_bootstrap_hook_for_test(fail_after_exclusive_lock)
    try:
        with pytest.raises(RuntimeError, match="forced bootstrap failure"):
            authority.initialize()
    finally:
        authority.close()

    database = root / AUTHORITY_RELATIVE_PATH
    with sqlite3.connect(database) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    assert tables == []
