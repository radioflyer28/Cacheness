"""Database-native compatibility-projection race contracts.

These schedules deliberately use independent metadata adapters.  A Python lock
inside one adapter cannot make either outcome deterministic.
"""

from __future__ import annotations

import threading
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from cacheness.custom_metadata import CacheMetadataLink
from cacheness.metadata import Base, ProjectionMutationResult, SqliteBackend
from cacheness.storage.backends.postgresql_backend import PostgresBackend


def _projection(locator: str) -> dict[str, object]:
    """Build the smallest complete compatibility projection row."""
    return {
        "description": "projection race",
        "data_type": "object",
        "prefix": "",
        "file_size": 1,
        "actual_path": locator,
        "metadata": {"actual_path": locator},
    }


def _run_concurrent_projection_mutations(
    first: SqliteBackend,
    second: SqliteBackend,
    *,
    expected_locator: str | None,
    first_locator: str,
    second_locator: str,
) -> list[ProjectionMutationResult]:
    """Release independent adapters only after both transactions are admitted."""
    admitted = threading.Barrier(2)
    results: list[ProjectionMutationResult] = []
    errors: list[BaseException] = []

    def pause(boundary: str, cache_key: str) -> None:
        # SQLite admits one writer at a time.  Synchronize immediately before
        # BEGIN IMMEDIATE, then let the database decide which writer enters.
        if boundary == "projection.transaction.before_writer" and cache_key == "projection-key":
            admitted.wait(timeout=5)

    first.set_projection_transaction_hook_for_test(pause)
    second.set_projection_transaction_hook_for_test(pause)

    def mutate(backend: SqliteBackend, locator: str) -> None:
        try:
            results.append(
                backend.conditional_projection_mutation(
                    "projection-key",
                    expected_locator=expected_locator,
                    replacement=_projection(locator),
                )
            )
        except BaseException as error:  # pragma: no cover - surfaced below.
            errors.append(error)

    first_thread = threading.Thread(target=mutate, args=(first, first_locator))
    second_thread = threading.Thread(target=mutate, args=(second, second_locator))
    first_thread.start()
    second_thread.start()
    first_thread.join(timeout=10)
    second_thread.join(timeout=10)
    first.set_projection_transaction_hook_for_test(None)
    second.set_projection_transaction_hook_for_test(None)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == [], [str(error) for error in errors]
    return results


def _mutate_from_independent_sqlite_process(
    database: str,
    expected_locator: str | None,
    replacement_locator: str,
    barrier,
    outcomes,
) -> None:
    """Run one projection mutation from a fresh spawned interpreter."""
    backend = SqliteBackend(database)

    def pause(boundary: str, cache_key: str) -> None:
        if boundary == "projection.transaction.before_writer" and cache_key == "process-proj-key":
            barrier.wait(timeout=10)

    try:
        backend.set_projection_transaction_hook_for_test(pause)
        result = backend.conditional_projection_mutation(
            "process-proj-key",
            expected_locator=expected_locator,
            replacement=_projection(replacement_locator),
        )
        outcomes.put(("result", result.status))
    except BaseException as error:  # pragma: no cover - asserted by the parent.
        outcomes.put(("error", type(error).__name__, str(error)))
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("expected_locator", "first_locator", "second_locator"),
    [
        (None, "/generation/m1-a", "/generation/m1-b"),
        ("/generation/m1", "/generation/m2-a", "/generation/m2-b"),
    ],
    ids=("absent", "existing"),
)
def test_sqlite_independent_adapters_have_one_atomic_projection_winner(
    tmp_path: Path,
    expected_locator: str | None,
    first_locator: str,
    second_locator: str,
) -> None:
    """An absent or M1 row changes once without leaking a database race error."""
    database = tmp_path / "projection.sqlite3"
    first = SqliteBackend(str(database))
    second = SqliteBackend(str(database))
    try:
        if expected_locator is not None:
            first.put_entry("projection-key", _projection(expected_locator))

        results = _run_concurrent_projection_mutations(
            first,
            second,
            expected_locator=expected_locator,
            first_locator=first_locator,
            second_locator=second_locator,
        )

        assert sorted(result.status for result in results) == ["applied", "mismatch"]
        current = first.get_entry("projection-key")
        assert current is not None
        assert current["metadata"]["actual_path"] in {first_locator, second_locator}
    finally:
        second.close()
        first.close()


@pytest.mark.parametrize(
    ("expected_locator", "first_locator", "second_locator"),
    [
        (None, "/generation/process-m1-a", "/generation/process-m1-b"),
        ("/generation/m1", "/generation/process-m2-a", "/generation/process-m2-b"),
    ],
    ids=("absent", "existing"),
)
def test_sqlite_spawned_processes_have_one_projection_winner(
    tmp_path: Path,
    expected_locator: str | None,
    first_locator: str,
    second_locator: str,
) -> None:
    """Separate Python processes prove the result is not an instance lock."""
    database = tmp_path / "projection-process.sqlite3"
    seed = SqliteBackend(str(database))
    try:
        if expected_locator is not None:
            seed.put_entry("process-proj-key", _projection(expected_locator))
    finally:
        seed.close()

    context = get_context("spawn")
    barrier = context.Barrier(2)
    outcomes = context.Queue()
    workers = [
        context.Process(
            target=_mutate_from_independent_sqlite_process,
            args=(
                str(database),
                expected_locator,
                locator,
                barrier,
                outcomes,
            ),
        )
        for locator in (first_locator, second_locator)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=20)
        assert worker.exitcode == 0

    result = [outcomes.get(timeout=5) for _ in workers]
    assert sorted(result) == [("result", "applied"), ("result", "mismatch")]


def test_sqlite_replacement_changes_link_ownership_only_for_winner(
    tmp_path: Path,
) -> None:
    """The winning M1-to-M2 transaction retires links; the mismatch is inert."""
    database = tmp_path / "projection-links.sqlite3"
    first = SqliteBackend(str(database))
    second = SqliteBackend(str(database))
    try:
        first.put_entry("projection-key", _projection("/generation/m1"))
        Base.metadata.create_all(first.engine, tables=[CacheMetadataLink.__table__])
        with first.SessionLocal() as session:
            session.add(
                CacheMetadataLink(
                    cache_key="projection-key",
                    metadata_table="custom_projection_test",
                    metadata_id=1,
                )
            )
            session.commit()

        results = _run_concurrent_projection_mutations(
            first,
            second,
            expected_locator="/generation/m1",
            first_locator="/generation/m2-a",
            second_locator="/generation/m2-b",
        )

        assert sorted(result.status for result in results) == ["applied", "mismatch"]
        with first.SessionLocal() as session:
            assert session.execute(select(CacheMetadataLink)).scalars().all() == []
        assert first.conditional_projection_mutation(
            "projection-key", expected_locator="/generation/m1", replacement=None
        ).status == "mismatch"
    finally:
        second.close()
        first.close()


def test_postgresql_projection_advisory_lock_is_bound_before_row_selection() -> None:
    """The absence guard is database-side and safe to compile for PostgreSQL."""
    commands: list[tuple[object, dict[str, object]]] = []

    class Session:
        bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        def execute(self, statement, params):
            commands.append((statement, params))

    PostgresBackend._acquire_projection_transaction_guard(Session(), "projection-key")

    statement, params = commands.pop()
    compiled = str(statement.compile(dialect=postgresql.dialect()))
    assert "pg_advisory_xact_lock" in compiled
    assert "hashtextextended" in compiled
    assert params == {"cache_key": "projection-key", "seed": 424242}
