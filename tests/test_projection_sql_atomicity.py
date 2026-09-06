"""Database-native compatibility-projection race contracts.

These schedules deliberately use independent metadata adapters.  A Python lock
inside one adapter cannot make either outcome deterministic.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from cacheness.metadata import ProjectionMutationResult, SqliteBackend


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
        if boundary == "projection.transaction.admitted" and cache_key == "projection-key":
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

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    return results


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
        winner = next(result.entry for result in results if result.applied)
        assert winner is not None
        current = first.get_entry("projection-key")
        assert current is not None
        assert current["actual_path"] == winner["actual_path"]
    finally:
        second.close()
        first.close()
