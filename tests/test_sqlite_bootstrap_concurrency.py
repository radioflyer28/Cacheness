"""Fresh-root SQLite authority bootstrap race contracts.

The schedules use independent authorities and spawned processes.  Their only
ordering seams are test-only bootstrap/lifecycle hooks; elapsed time is never
used to decide a successful interleaving.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import threading

import pytest

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationRequiredError,
)
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.lifecycle_authority import EntryExpectation, MutationSpec
from cacheness.storage.sqlite_lifecycle_authority import (
    AUTHORITY_RELATIVE_PATH,
    SqliteLifecycleAuthority,
)


_CLASSIFIED = "authority.bootstrap.classified"
_BEFORE_PROMOTION = "put.before_promotion"


def _bootstrap_spec(key: str) -> MutationSpec:
    """Build a bounded direct authority mutation used only to bootstrap a store."""
    return MutationSpec.create(
        operation_id=f"bootstrap-operation-{key}",
        key=f"bootstrap-{key}",
        generation=f"bootstrap-generation-{key}",
        candidate_locator=f"generations/bootstrap-{key}.native",
        expected=EntryExpectation.absent(),
    )


def _bootstrap_then_put(
    root: str,
    *,
    key: str,
    value: str,
    bootstrap_barrier,
    outcomes,
    promotion_barrier=None,
) -> None:
    """Bootstrap one independent authority, then exercise a real BlobStore put."""
    authority = SqliteLifecycleAuthority.for_root(root)
    store: BlobStore | None = None
    try:
        def pause_after_missing_classification(boundary: str) -> None:
            if boundary == _CLASSIFIED:
                bootstrap_barrier.wait(timeout=20)

        authority.set_bootstrap_hook_for_test(pause_after_missing_classification)
        prepared = authority.prepare_mutation(_bootstrap_spec(key))
        authority.abort_mutation(prepared, candidate_persisted=False)
        store = BlobStore(root, backend="json", lifecycle_authority=authority)
        if promotion_barrier is not None:
            def pause_before_promotion(boundary: str) -> None:
                if boundary == _BEFORE_PROMOTION:
                    promotion_barrier.wait(timeout=20)

            store.lifecycle.test_hook = pause_before_promotion
        store.put(value, key=key)
        outcomes.put(("ok", key, value))
    except CacheBlobLifecycleConflictError:
        outcomes.put(("conflict", key, value))
    except BaseException as error:  # pragma: no cover - reported by parent assertions.
        outcomes.put(("error", type(error).__name__, str(error)))
    finally:
        if store is not None:
            store.close()
        authority.close()


def _threaded_bootstrap_worker(
    root: str,
    key: str,
    value: str,
    bootstrap_barrier: threading.Barrier,
    outcomes: multiprocessing.Queue,
) -> None:
    """Expose the two-instance race without sharing an authority instance."""
    _bootstrap_then_put(
        root,
        key=key,
        value=value,
        bootstrap_barrier=bootstrap_barrier,
        outcomes=outcomes,
    )


def _join(thread: threading.Thread) -> None:
    """Join a deterministic test worker without accepting a deadlock."""
    thread.join(timeout=20)
    assert not thread.is_alive(), "bootstrap worker did not finish"


def test_independent_authority_instances_join_one_fresh_root_then_commit_distinct_keys(
    tmp_path: Path,
) -> None:
    """Two first mutators reclassify a valid winner instead of leaking mkdir races."""
    root = tmp_path / "threaded-fresh-root"
    bootstrap_barrier = threading.Barrier(2)
    outcomes: multiprocessing.Queue = multiprocessing.Queue()
    workers = [
        threading.Thread(
            target=_threaded_bootstrap_worker,
            args=(str(root), key, value, bootstrap_barrier, outcomes),
        )
        for key, value in (("thread-left", "left"), ("thread-right", "right"))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        _join(worker)

    assert sorted(outcomes.get(timeout=5) for _ in workers) == [
        ("ok", "thread-left", "left"),
        ("ok", "thread-right", "right"),
    ]
    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.get("thread-left") == "left"
        assert reopened.get("thread-right") == "right"
    finally:
        reopened.close()


def test_spawned_fresh_authorities_join_one_root_and_commit_distinct_keys(
    tmp_path: Path,
) -> None:
    """Separate spawned interpreters prove bootstrap is not an instance-lock race."""
    root = tmp_path / "spawned-distinct-root"
    context = multiprocessing.get_context("spawn")
    bootstrap_barrier = context.Barrier(2)
    outcomes = context.Queue()
    workers = [
        context.Process(
            target=_bootstrap_then_put,
            kwargs={
                "root": str(root),
                "key": key,
                "value": value,
                "bootstrap_barrier": bootstrap_barrier,
                "outcomes": outcomes,
            },
        )
        for key, value in (("process-left", "left"), ("process-right", "right"))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    assert sorted(outcomes.get(timeout=5) for _ in workers) == [
        ("ok", "process-left", "left"),
        ("ok", "process-right", "right"),
    ]
    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.get("process-left") == "left"
        assert reopened.get("process-right") == "right"
    finally:
        reopened.close()


def test_spawned_fresh_authorities_keep_same_key_cas_deterministic(
    tmp_path: Path,
) -> None:
    """A fresh-root bootstrap join does not weaken expected-generation CAS."""
    root = tmp_path / "spawned-same-key-root"
    context = multiprocessing.get_context("spawn")
    bootstrap_barrier = context.Barrier(2)
    promotion_barrier = context.Barrier(2)
    outcomes = context.Queue()
    workers = [
        context.Process(
            target=_bootstrap_then_put,
            kwargs={
                "root": str(root),
                "key": "same-key",
                "value": value,
                "bootstrap_barrier": bootstrap_barrier,
                "promotion_barrier": promotion_barrier,
                "outcomes": outcomes,
            },
        )
        for value in ("left", "right")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    result = sorted(outcomes.get(timeout=5) for _ in workers)
    assert [outcome[0] for outcome in result] == ["conflict", "ok"]
    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.get("same-key") in {"left", "right"}
    finally:
        reopened.close()


def test_late_regular_leaf_reclassifies_instead_of_rejecting_valid_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leaf created after the final check joins under the original deadline."""
    root = tmp_path / "late-authority-leaf"
    reserved = root / AUTHORITY_RELATIVE_PATH.parent
    database = root / AUTHORITY_RELATIVE_PATH
    root.mkdir()
    reserved.mkdir()
    temporary = reserved / "inflight-observation"
    temporary.write_bytes(b"temporary observer state")

    authority = SqliteLifecycleAuthority.for_root(root)
    original_classify = authority._classify_for_open
    classifications = 0

    def classify_with_late_leaf() -> str:
        nonlocal classifications
        state = original_classify()
        classifications += 1
        if classifications == 5:
            assert state == "established"
            temporary.unlink()
            descriptor = os.open(database, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
        return state

    monkeypatch.setattr(authority, "_classify_for_open", classify_with_late_leaf)
    try:
        prepared = authority.prepare_mutation(_bootstrap_spec("late-leaf"))
        assert database.is_file()
        authority.abort_mutation(prepared, candidate_persisted=False)
    finally:
        authority.close()


def test_awaited_bootstrap_reclassifies_a_leaf_that_appears_during_namespace_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded join rechecks a valid leaf before rejecting established state."""
    root = tmp_path / "awaited-authority-leaf"
    reserved = root / AUTHORITY_RELATIVE_PATH.parent
    database = root / AUTHORITY_RELATIVE_PATH
    root.mkdir()
    reserved.mkdir()
    temporary = reserved / "inflight-observation"
    temporary.write_bytes(b"temporary observer state")

    authority = SqliteLifecycleAuthority.for_root(root)
    original_classify = authority._classify_for_open
    classifications = 0

    def classify_with_leaf_during_namespace_scan() -> str:
        nonlocal classifications
        state = original_classify()
        classifications += 1
        if classifications == 6:
            assert state == "established"
            temporary.unlink()
            descriptor = os.open(database, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
        return state

    monkeypatch.setattr(authority, "_classify_for_open", classify_with_leaf_during_namespace_scan)
    try:
        prepared = authority.prepare_mutation(_bootstrap_spec("awaited-leaf"))
        assert database.is_file()
        authority.abort_mutation(prepared, candidate_persisted=False)
    finally:
        authority.close()


@pytest.mark.parametrize(
    "substitution",
    ("root-file", "reserved-file", "database-file", "root-symlink", "reserved-symlink", "database-symlink"),
)
def test_bootstrap_rejects_wrong_winner_objects_unchanged(
    tmp_path: Path,
    substitution: str,
) -> None:
    """A loser never adopts or mutates hostile bootstrap evidence after mkdir loses."""
    if substitution.endswith("symlink") and not hasattr(os, "symlink"):
        pytest.skip("symlink support is unavailable")

    root = tmp_path / substitution
    reserved = root / AUTHORITY_RELATIVE_PATH.parent
    database = root / AUTHORITY_RELATIVE_PATH
    target = tmp_path / f"{substitution}-target"
    captured: dict[str, object] = {}

    def install_substitution(boundary: str) -> None:
        if boundary != _CLASSIFIED:
            return
        if substitution == "root-file":
            root.write_bytes(b"hostile root")
            captured["path"] = root
            captured["bytes"] = root.read_bytes()
        elif substitution == "reserved-file":
            root.mkdir()
            reserved.write_bytes(b"hostile reserved")
            captured["path"] = reserved
            captured["bytes"] = reserved.read_bytes()
        elif substitution == "database-file":
            root.mkdir()
            reserved.mkdir()
            database.write_bytes(b"hostile database")
            captured["path"] = database
            captured["bytes"] = database.read_bytes()
        elif substitution == "root-symlink":
            target.mkdir()
            root.symlink_to(target, target_is_directory=True)
            captured["path"] = root
            captured["target"] = os.readlink(root)
        elif substitution == "reserved-symlink":
            root.mkdir()
            target.mkdir()
            reserved.symlink_to(target, target_is_directory=True)
            captured["path"] = reserved
            captured["target"] = os.readlink(reserved)
        else:
            root.mkdir()
            reserved.mkdir()
            target.write_bytes(b"hostile target")
            database.symlink_to(target)
            captured["path"] = database
            captured["target"] = os.readlink(database)

    authority = SqliteLifecycleAuthority.for_root(root)
    authority.set_bootstrap_hook_for_test(install_substitution)
    try:
        with pytest.raises(CacheBlobMigrationRequiredError):
            authority.prepare_mutation(_bootstrap_spec(substitution))
    finally:
        authority.close()

    hostile_path = captured["path"]
    assert isinstance(hostile_path, Path)
    if "bytes" in captured:
        assert hostile_path.read_bytes() == captured["bytes"]
    else:
        assert hostile_path.is_symlink()
        assert os.readlink(hostile_path) == captured["target"]
    if substitution.endswith("symlink"):
        assert not target.is_dir() or tuple(target.iterdir()) == ()
