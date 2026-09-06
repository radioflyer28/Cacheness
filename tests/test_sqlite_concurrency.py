"""SQLite lifecycle contention outcomes are safe, bounded, and recoverable."""

from __future__ import annotations

import math
from pathlib import Path
import sqlite3
from threading import Barrier, Lock, Thread

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
)
from cacheness.storage.lifecycle_authority import EntryExpectation, MutationSpec
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


def _is_retryable_timeout(error: BaseException) -> bool:
    """Recognize the only allowed bounded SQLite contention outcome."""
    if not isinstance(error, CacheBlobLifecycleTimeoutError):
        return False
    context = error.context
    cause = error.__cause__
    code = getattr(cause, "sqlite_errorcode", None)
    return (
        context.get("operation") == "lifecycle_authority"
        and context.get("retryable") is True
        and isinstance(context.get("authority_path"), str)
        and bool(context["authority_path"])
        and isinstance(context.get("authority_busy_timeout_seconds"), (int, float))
        and not isinstance(context["authority_busy_timeout_seconds"], bool)
        and math.isfinite(context["authority_busy_timeout_seconds"])
        and context["authority_busy_timeout_seconds"] > 0
        and isinstance(cause, sqlite3.OperationalError)
        and type(code) is int
        and (code & 0xFF) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
    )


def _spec(key: str, operation_id: str) -> MutationSpec:
    """Build one explicit authority mutation without publishing a payload."""
    return MutationSpec.create(
        operation_id=operation_id,
        key=key,
        generation=f"generation-{operation_id}",
        candidate_locator=f"generations/{operation_id}.native",
        expected=EntryExpectation.absent(),
        manifest=f"manifest-{operation_id}".encode("utf-8"),
    )


def test_sqlite_workers_account_for_safe_contention_outcomes(tmp_path: Path) -> None:
    """Every worker completes, conflicts, or returns a contextual retryable timeout."""
    authority = SqliteLifecycleAuthority.for_root(tmp_path / "authority")
    authority.begin_clear()
    barrier = Barrier(4)
    records: list[dict[str, int]] = []
    successful_keys: set[str] = set()
    records_lock = Lock()

    def attempt(worker: int, index: int, key: str, record: dict[str, int]) -> None:
        prepared = None
        record["attempted"] += 1
        try:
            prepared = authority.prepare_mutation(_spec(key, f"{worker}-{index}"))
            authority.record_verification_for_test(prepared)
            authority.promote_mutation(prepared)
        except CacheBlobLifecycleConflictError:
            if prepared is not None:
                authority.abort_mutation(prepared)
            record["conflicts"] += 1
        except CacheBlobLifecycleTimeoutError as error:
            if not _is_retryable_timeout(error):
                record["hard_failures"] += 1
            else:
                record["timeouts"] += 1
        except Exception:  # pragma: no cover - assertion below records any regression.
            record["hard_failures"] += 1
        else:
            record["successes"] += 1
            with records_lock:
                successful_keys.add(key)

    def worker(worker: int) -> None:
        record = {"attempted": 0, "successes": 0, "conflicts": 0, "timeouts": 0, "hard_failures": 0}
        barrier.wait(timeout=5)
        attempt(worker, 0, "shared-key", record)
        attempt(worker, 1, f"worker-{worker}-one", record)
        attempt(worker, 2, f"worker-{worker}-two", record)
        with records_lock:
            records.append(record)

    threads = [Thread(target=worker, args=(worker_id,)) for worker_id in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert len(records) == 4
    assert all(record["hard_failures"] == 0 for record in records)
    assert all(
        record["successes"] + record["conflicts"] + record["timeouts"] == record["attempted"]
        for record in records
    )
    assert all(
        record["successes"] + record["conflicts"] + record["timeouts"] > 0
        for record in records
    )
    assert len(successful_keys) == 9
    assert {entry.key for entry in authority.list_entries()} == successful_keys
    assert authority.pending_mutations() == ()
    assert authority.pending_cleanup_debts() == ()
    authority.close()

    reopened = SqliteLifecycleAuthority.for_root(tmp_path / "authority")
    try:
        assert {entry.key for entry in reopened.list_entries()} == successful_keys
        assert reopened.pending_mutations() == ()
        assert reopened.pending_cleanup_debts() == ()
    finally:
        reopened.close()
