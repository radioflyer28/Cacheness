"""Deterministic admission coverage for short SQLite authority writes."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


def test_warmed_authorities_share_fifo_admission_and_retire_gate(tmp_path: Path) -> None:
    """A paused eligible writer cannot be overtaken by another authority instance."""
    first = SqliteLifecycleAuthority.for_root(tmp_path)
    second = SqliteLifecycleAuthority.for_root(tmp_path)
    first.begin_clear()  # Materialize and validate the bootstrap/schema path before admission.

    first_eligible = Event()
    second_enqueued = Event()
    release_first = Event()
    errors: list[BaseException] = []
    grants: list[str] = []

    def first_observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.eligible":
            first_eligible.set()
            assert release_first.wait(timeout=2)
        if event == "writer_admission.granted":
            grants.append("first")

    def second_observer(event: str, _timestamp: float) -> None:
        if event == "writer_admission.enqueued":
            second_enqueued.set()
        if event == "writer_admission.granted":
            grants.append("second")

    first.set_admission_observer_for_test(first_observer)
    second.set_admission_observer_for_test(second_observer)

    def run(authority: SqliteLifecycleAuthority) -> None:
        try:
            authority.begin_clear()
        except BaseException as error:  # pragma: no cover - asserted after joins
            errors.append(error)

    first_thread = Thread(target=run, args=(first,))
    second_thread = Thread(target=run, args=(second,))
    try:
        first_thread.start()
        assert first_eligible.wait(timeout=2)
        second_thread.start()
        assert second_enqueued.wait(timeout=2)
        release_first.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert errors == []
        assert grants == ["first", "second"]
    finally:
        release_first.set()
        first.close()
        second.close()

    assert SqliteLifecycleAuthority.admission_registry_size_for_test() == 0
