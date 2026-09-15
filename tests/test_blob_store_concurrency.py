"""Deterministic direct BlobStore coordination and read-race contracts."""

from __future__ import annotations

from contextlib import contextmanager
import json
import multiprocessing
import threading
from pathlib import Path

import pytest

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobRecoverableCleanupError,
)
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.coordination import KeyCoordinatorRegistry
from _lifecycle_test_support import ReleaseGate


def _store(root: Path) -> BlobStore:
    """Create the supported local topology for progress and safety tests."""
    return BlobStore(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        ),
        cache_dir=root,
    )


LIFECYCLE_BASELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "lifecycle_authority_baseline.json"
)


def test_overlap_release_envelope_is_benchmark_only() -> None:
    """Overlap evidence is auditable without becoming a runtime policy knob."""
    with LIFECYCLE_BASELINE_PATH.open(encoding="utf-8") as handle:
        baseline = json.load(handle)

    p05 = baseline["metrics"]["distinct_key_overlap"]["ratio"]["p05"]
    multiplier = baseline["multipliers"]["distinct_key_overlap"]
    envelope = baseline["derived"]["release_envelopes"][
        "distinct_key_overlap_min_ratio"
    ]

    assert envelope == p05 * multiplier
    assert "distinct_key_overlap" not in baseline["derived"]["configuration"]


def _put_from_independent_process(
    root: str,
    started: multiprocessing.synchronize.Event,
    completed: multiprocessing.synchronize.Event,
    errors: multiprocessing.queues.Queue,
) -> None:
    """Exercise public admission from a fresh process, not a shared lock map."""
    started.set()
    store = _store(root)
    try:
        store.put("post-snapshot", key="post-snapshot")
        completed.set()
    except BaseException as exc:  # pragma: no cover - surfaced in parent.
        errors.put(repr(exc))
    finally:
        store.close()


def _put_from_ready_independent_process(
    root: str,
    ready: multiprocessing.synchronize.Event,
    run: multiprocessing.synchronize.Event,
    completed: multiprocessing.synchronize.Event,
    errors: multiprocessing.queues.Queue,
) -> None:
    """Open independently before a scheduler race, then perform one public put."""
    store = _store(root)
    try:
        ready.set()
        assert run.wait(timeout=15)
        store.put("child", key="child-key")
        completed.set()
    except BaseException as exc:  # pragma: no cover - surfaced in parent.
        errors.put(repr(exc))
    finally:
        store.close()


def _join(thread: threading.Thread) -> None:
    """Join one deterministic test worker without hiding a deadlock."""
    thread.join(timeout=5)
    assert not thread.is_alive(), "worker did not finish within the bounded wait"


def test_key_registry_retires_entries_after_exception_and_high_cardinality():
    """Refcounted entries are removed after every normal and exceptional holder."""
    registry = KeyCoordinatorRegistry()

    with pytest.raises(RuntimeError, match="intentional"):
        with registry.hold("exception-key"):
            assert registry.size == 1
            raise RuntimeError("intentional")
    assert registry.size == 0

    for number in range(128):
        with registry.hold(f"key-{number}"):
            assert registry.size == 1
    assert registry.size == 0


def test_key_registry_sorts_multi_key_acquisition_without_deadlock():
    """Inverted callers use one stable order and complete under bounded joins."""
    registry = KeyCoordinatorRegistry()
    first_entered = threading.Event()
    release_first = threading.Event()
    errors: list[BaseException] = []

    def first() -> None:
        try:
            with registry.hold_many(("z-key", "a-key")):
                first_entered.set()
                assert release_first.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - re-raised below.
            errors.append(exc)

    def second() -> None:
        try:
            with registry.hold_many(("a-key", "z-key")):
                pass
        except BaseException as exc:  # pragma: no cover - re-raised below.
            errors.append(exc)

    first_thread = threading.Thread(target=first)
    first_thread.start()
    assert first_entered.wait(timeout=5)
    second_thread = threading.Thread(target=second)
    second_thread.start()
    release_first.set()
    _join(first_thread)
    _join(second_thread)
    assert errors == []
    assert registry.size == 0


def test_independent_write_write_race_has_one_cas_winner(tmp_path):
    """Independent stores prove CAS, not the local lock, selects the winner."""
    root = tmp_path / "write-write"
    first = _store(root)
    second = _store(root)
    first.initialize()
    barrier = threading.Barrier(2)
    results: list[str] = []
    errors: list[BaseException] = []

    def pause_before_publish(seam, _record) -> None:
        if seam == "manifest_publish":
            barrier.wait(timeout=5)

    first.lifecycle.fault_hook = pause_before_publish
    second.lifecycle.fault_hook = pause_before_publish

    def write(store: BlobStore, value: str) -> None:
        try:
            results.append(store.put(value, key="same-key"))
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)

    first_thread = threading.Thread(target=write, args=(first, "first"))
    second_thread = threading.Thread(target=write, args=(second, "second"))
    try:
        first_thread.start()
        second_thread.start()
        _join(first_thread)
        _join(second_thread)

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], CacheBlobLifecycleConflictError)
        assert first.get("same-key") in {"first", "second"}
    finally:
        second.close()
        first.close()


def test_independent_write_delete_race_has_one_cas_winner(tmp_path):
    """A write and delete from independent stores cannot both replace authority."""
    root = tmp_path / "write-delete"
    seed = _store(root)
    key = "same-key"
    try:
        seed.put("original", key=key)
    finally:
        seed.close()

    writer = _store(root)
    deleter = _store(root)
    barrier = threading.Barrier(2)
    results: list[tuple[str, object]] = []
    errors: list[BaseException] = []

    def pause_before_authority(seam, _record) -> None:
        if seam in {"manifest_publish", "tombstone_publish"}:
            barrier.wait(timeout=5)

    writer.lifecycle.fault_hook = pause_before_authority
    deleter.lifecycle.fault_hook = pause_before_authority

    def overwrite() -> None:
        try:
            results.append(("write", writer.put("replacement", key=key)))
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)

    def delete() -> None:
        try:
            results.append(("delete", deleter.delete(key)))
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)

    writer_thread = threading.Thread(target=overwrite)
    deleter_thread = threading.Thread(target=delete)
    try:
        writer_thread.start()
        deleter_thread.start()
        _join(writer_thread)
        _join(deleter_thread)

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], CacheBlobLifecycleConflictError)
        assert writer.get(key) in {None, "replacement"}
    finally:
        deleter.close()
        writer.close()


def test_distinct_key_put_completes_while_another_key_is_pre_cas(tmp_path):
    """A per-key holder never serializes an unrelated ordinary write."""
    store = _store(tmp_path / "distinct")
    key_a_entered = threading.Event()
    release_key_a = threading.Event()
    key_b_finished = threading.Event()
    errors: list[BaseException] = []

    def pause_key_a(seam, record) -> None:
        if seam == "manifest_publish" and record.key == "key-a":
            key_a_entered.set()
            assert release_key_a.wait(timeout=5)

    store.lifecycle.fault_hook = pause_key_a

    def put(key: str, value: str, completed: threading.Event | None = None) -> None:
        try:
            store.put(value, key=key)
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)
        finally:
            if completed is not None:
                completed.set()

    key_a_thread = threading.Thread(target=put, args=("key-a", "a"))
    key_a_thread.start()
    assert key_a_entered.wait(timeout=5)
    key_b_thread = threading.Thread(target=put, args=("key-b", "b", key_b_finished))
    key_b_thread.start()
    try:
        assert key_b_finished.wait(timeout=5)
        assert store.get("key-b") == "b"
    finally:
        release_key_a.set()
        _join(key_a_thread)
        _join(key_b_thread)
        store.close()
    assert errors == []


def _run_exact_snapshot_clear_delete_collision(
    root: Path,
) -> tuple[int, bool | CacheBlobLifecycleConflictError]:
    """Exercise one bounded clear/delete collision and assert its safe terminal state."""
    store = _store(root)
    gate = ReleaseGate()
    tombstone_promotion = threading.Barrier(2)
    clear_results: list[int] = []
    clear_errors: list[BaseException] = []
    delete_results: list[bool] = []
    delete_errors: list[BaseException] = []
    clearer: threading.Thread | None = None
    deleter: threading.Thread | None = None

    def pause_after_snapshot(boundary: str) -> None:
        if boundary == "clear.snapshot_committed":
            gate.wait_at_boundary()

    def pause_before_tombstone_promotion(seam: str, _record: object) -> None:
        if seam in {"delete.before_tombstone_promotion", "tombstone_publish"}:
            tombstone_promotion.wait(timeout=5)

    def clear() -> None:
        try:
            clear_results.append(store.clear())
        except BaseException as error:  # pragma: no cover - re-raised below.
            clear_errors.append(error)

    def delete() -> None:
        try:
            delete_results.append(store.delete("snapshot-key"))
        except BaseException as error:  # pragma: no cover - re-raised below.
            delete_errors.append(error)

    try:
        store.put("present", key="snapshot-key")
        store.lifecycle.test_hook = pause_after_snapshot
        store.lifecycle.fault_hook = pause_before_tombstone_promotion
        clearer = threading.Thread(target=clear, name="exact-snapshot-clear")
        clearer.start()
        assert gate.arrived.wait(timeout=5)

        # Clear's short admission closes only until this exact snapshot commits.
        # Release it before starting the public delete, then make both tombstone
        # promotions meet at the lifecycle seam without timing assumptions.
        gate.release()
        deleter = threading.Thread(target=delete, name="exact-snapshot-delete")
        deleter.start()
        _join(clearer)
        _join(deleter)

        assert clear_errors == []
        assert len(clear_results) == 1
        assert clear_results[0] in {0, 1}

        assert len(delete_results) + len(delete_errors) == 1
        if delete_errors:
            assert len(delete_errors) == 1
            assert isinstance(delete_errors[0], CacheBlobLifecycleConflictError)
            delete_outcome: bool | CacheBlobLifecycleConflictError = delete_errors[0]
        else:
            assert len(delete_results) == 1
            assert isinstance(delete_results[0], bool)
            delete_outcome = delete_results[0]

        assert store.get("snapshot-key") is None
        assert store.lifecycle_authority.read_entry("snapshot-key") is None
        assert store.lifecycle_authority.pending_cleanup_debts() == ()
        return clear_results[0], delete_outcome
    finally:
        gate.release()
        tombstone_promotion.abort()
        try:
            if clearer is not None:
                _join(clearer)
            if deleter is not None:
                _join(deleter)
        finally:
            store.close()


def test_clear_and_delete_converge_after_an_exact_snapshot(tmp_path: Path) -> None:
    """A paused clear and public delete accept only safe bounded outcomes."""
    _run_exact_snapshot_clear_delete_collision(tmp_path / "clear-delete")


def test_clear_and_delete_exact_snapshot_stress_preserves_safety_across_valid_outcomes(
    tmp_path: Path,
) -> None:
    """Sixteen isolated exact-snapshot collisions preserve the same safety oracle."""
    for case in range(16):
        _run_exact_snapshot_clear_delete_collision(tmp_path / f"clear-delete-{case}")


def test_reconciliation_and_mutation_converge_on_distinct_indexed_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconciliation may retire old debt while a later mutation is paused."""
    store = _store(tmp_path / "reconcile-mutation")
    gate = ReleaseGate()
    errors: list[BaseException] = []

    def pause_before_promotion(boundary: str) -> None:
        if boundary == "put.before_promotion":
            gate.wait_at_boundary()

    def promote_latest() -> None:
        try:
            store.put("latest", key="race-key")
        except BaseException as error:  # pragma: no cover - re-raised below.
            errors.append(error)

    try:
        store.put("old", key="race-key")
        original_delete = store._delete_or_prove_absent
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            lambda _locator: (_ for _ in ()).throw(OSError("defer old cleanup")),
        )
        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.put("middle", key="race-key")
        monkeypatch.setattr(store, "_delete_or_prove_absent", original_delete)

        store.lifecycle.test_hook = pause_before_promotion
        mutator = threading.Thread(target=promote_latest)
        mutator.start()
        assert gate.arrived.wait(timeout=5)

        report = store.reconcile(apply=True)
        gate.release()
        _join(mutator)

        assert len(errors) == 1
        assert isinstance(errors[0], CacheBlobLifecycleConflictError)
        assert report.applied is True
        assert store.get("race-key") == "middle"
        assert store.lifecycle_authority.pending_cleanup_debts() == ()
    finally:
        gate.release()
        store.close()


def _retired_scheduler_clear_snapshot_does_not_delete_a_post_snapshot_key(tmp_path):
    """Clear's finite target inventory excludes a key committed after snapshot."""
    store = _store(tmp_path / "clear-post-snapshot")
    snapshot_complete = threading.Event()
    release_clear = threading.Event()
    post_snapshot_done = threading.Event()
    errors: list[BaseException] = []
    try:
        store.put("before", key="before-key")

        def pause_after_snapshot(seam, _record) -> None:
            if seam == "clear_snapshot_complete":
                snapshot_complete.set()
                assert release_clear.wait(timeout=5)

        store.lifecycle.fault_hook = pause_after_snapshot

        def clear() -> None:
            try:
                assert store.clear() == 1
            except BaseException as exc:  # pragma: no cover - asserted below.
                errors.append(exc)

        def put_after_snapshot() -> None:
            try:
                store.put("after", key="after-key")
            except BaseException as exc:  # pragma: no cover - asserted below.
                errors.append(exc)
            finally:
                post_snapshot_done.set()

        clear_thread = threading.Thread(target=clear)
        clear_thread.start()
        assert snapshot_complete.wait(timeout=5)
        put_thread = threading.Thread(target=put_after_snapshot)
        put_thread.start()
        release_clear.set()
        assert post_snapshot_done.wait(timeout=5)
        _join(clear_thread)
        _join(put_thread)

        assert errors == []
        assert store.get("before-key") is None
        assert store.get("after-key") == "after"
    finally:
        store.close()


def test_distinct_key_put_completes_during_paused_authority_promotion(
    tmp_path: Path,
) -> None:
    """A paused key's promotion cannot serialize another key's payload work."""
    root = tmp_path / "authority-promotion"
    store = _store(root)
    entered = threading.Event()
    release = threading.Event()
    second_done = threading.Event()
    errors: list[BaseException] = []

    def pause_before_promotion(seam: str, record: object) -> None:
        if seam == "manifest_publish" and record.key == "key-a":
            entered.set()
            assert release.wait(timeout=5)

    store.lifecycle.fault_hook = pause_before_promotion

    def put(key: str, value: str, done: threading.Event | None = None) -> None:
        try:
            store.put(value, key=key)
            if done is not None:
                done.set()
        except BaseException as exc:  # pragma: no cover - asserted by caller.
            errors.append(exc)

    first = threading.Thread(target=put, args=("key-a", "value-a"))
    second = threading.Thread(target=put, args=("key-b", "value-b", second_done))
    try:
        first.start()
        assert entered.wait(timeout=5)
        second.start()
        assert second_done.wait(timeout=5)
        release.set()
        _join(first)
        _join(second)
        assert errors == []
        assert store.get("key-a") == "value-a"
        assert store.get("key-b") == "value-b"
    finally:
        release.set()
        _join(first)
        _join(second)
        store.close()

    reopened = _store(root)
    try:
        assert reopened.get("key-a") == "value-a"
        assert reopened.get("key-b") == "value-b"
    finally:
        reopened.close()


def test_independent_process_put_completes_during_paused_authority_promotion(
    tmp_path: Path,
) -> None:
    """SQLite-only promotion does not hold a store lease across payload work."""
    root = tmp_path / "authority-process-promotion"
    store = _store(root)
    store.put("seed", key="seed-key")
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def pause_before_promotion(seam: str, record: object) -> None:
        if seam == "manifest_publish" and record.key == "parent-key":
            entered.set()
            assert release.wait(timeout=30)

    store.lifecycle.fault_hook = pause_before_promotion

    def first_put() -> None:
        try:
            store.put("parent", key="parent-key")
        except BaseException as exc:  # pragma: no cover - asserted by parent.
            errors.append(exc)

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    run = context.Event()
    completed = context.Event()
    child_errors = context.Queue()
    first = threading.Thread(target=first_put)
    child = context.Process(
        target=_put_from_ready_independent_process,
        args=(str(root), ready, run, completed, child_errors),
    )
    try:
        child.start()
        assert ready.wait(timeout=15)
        first.start()
        assert entered.wait(timeout=15)
        run.set()
        assert completed.wait(timeout=15)
        child.join(timeout=15)
        assert child.exitcode == 0
        assert child_errors.empty()
        release.set()
        run.set()
        _join(first)
        assert errors == []
    finally:
        release.set()
        _join(first)
        if child.is_alive():
            child.terminate()
        child.join(timeout=15)
        store.close()

    reopened = _store(root)
    try:
        assert reopened.get("parent-key") == "parent"
        assert reopened.get("child-key") == "child"
    finally:
        reopened.close()


def _retired_scheduler_live_clear_transition_lease_preserves_creator_return_count(
    tmp_path: Path,
) -> None:
    """Constructor recovery waits for a live clearer after snapshot admission ends."""
    root = tmp_path / "live-clear-transition-lease"
    owner = _store(root)
    snapshot_released = threading.Event()
    allow_creator_continue = threading.Event()
    constructor_finished = threading.Event()
    clear_result: list[int] = []
    errors: list[BaseException] = []

    def pause_after_admission_release(seam: str, _record: object) -> None:
        if seam == "clear_snapshot_admission_released":
            snapshot_released.set()
            assert allow_creator_continue.wait(timeout=5)

    def clear_owner() -> None:
        try:
            clear_result.append(owner.clear())
        except BaseException as exc:  # pragma: no cover - asserted by parent.
            errors.append(exc)

    def reopen_during_live_clear() -> None:
        reopened: BlobStore | None = None
        try:
            reopened = _store(root)
        except BaseException as exc:  # pragma: no cover - asserted by parent.
            errors.append(exc)
        finally:
            if reopened is not None:
                reopened.close()
            constructor_finished.set()

    try:
        owner.put("present", key="present")
        owner.lifecycle.fault_hook = pause_after_admission_release
        clearer = threading.Thread(target=clear_owner)
        clearer.start()
        assert snapshot_released.wait(timeout=5)

        reopening = threading.Thread(target=reopen_during_live_clear)
        reopening.start()
        # Reopen may pass aggregate admission, but it must remain blocked on
        # the already-held clear continuation lease rather than consume the
        # record and changing the initiating ``clear()`` result.
        assert not constructor_finished.wait(timeout=0.2)

        allow_creator_continue.set()
        _join(clearer)
        _join(reopening)
        assert clear_result == [1]
        assert errors == []
    finally:
        allow_creator_continue.set()
        owner.close()


def _retired_scheduler_clear_snapshot_excludes_a_later_independent_process_write(
    tmp_path,
):
    """Cross-process admission holds the exact clear snapshot stable."""
    root = tmp_path / "cross-process-clear-admission"
    store = _store(root)
    snapshot_complete = threading.Event()
    release_snapshot = threading.Event()
    clear_errors: list[BaseException] = []
    clear_result: list[int] = []

    def pause_after_snapshot(seam, _record) -> None:
        if seam == "clear_snapshot_complete":
            snapshot_complete.set()
            assert release_snapshot.wait(timeout=10)

    def clear() -> None:
        try:
            clear_result.append(store.clear())
        except BaseException as exc:  # pragma: no cover - asserted below.
            clear_errors.append(exc)

    try:
        store.put("present", key="present-before-snapshot")
        store.lifecycle.fault_hook = pause_after_snapshot
        clearer = threading.Thread(target=clear)
        clearer.start()
        assert snapshot_complete.wait(timeout=10)

        context = multiprocessing.get_context("spawn")
        child_started = context.Event()
        child_completed = context.Event()
        child_errors = context.Queue()
        writer = context.Process(
            target=_put_from_independent_process,
            args=(str(root), child_started, child_completed, child_errors),
        )
        writer.start()
        assert child_started.wait(timeout=10)
        # The independent public put cannot pass its shared OS admission while
        # the clear still owns exclusive snapshot admission.
        assert not child_completed.wait(timeout=0.2)

        release_snapshot.set()
        _join(clearer)
        writer.join(timeout=10)
        assert writer.exitcode == 0
        assert child_errors.empty()
        assert clear_errors == []
        assert clear_result == [1]
        assert store.get("present-before-snapshot") is None
        assert store.get("post-snapshot") == "post-snapshot"
    finally:
        release_snapshot.set()
        store.close()


def _authority_exists_reacquires_once_only_after_an_independent_generation_change(
    tmp_path, monkeypatch
):
    """Existence checks use M1/snapshot/M2 rather than a stale path assertion."""
    root = tmp_path / "exists-generation-retry"
    reader = _store(root)
    writer = _store(root)
    key = "same-key"
    try:
        reader.put("old", key=key)
        original_read_entry = reader.lifecycle_authority.read_entry
        reads = 0

        def change_after_snapshot(blob_key: str):
            nonlocal reads
            reads += 1
            if reads == 2:
                writer.put("new", key=key)
            return original_read_entry(blob_key)

        monkeypatch.setattr(
            reader.lifecycle_authority,
            "read_entry",
            change_after_snapshot,
        )
        assert reader.exists(key) is True
        # M1/M2 then exactly one retry's M1/M2; no unbounded retry loop.
        assert reads == 4
        assert reader.get(key) == "new"
    finally:
        writer.close()
        reader.close()


def test_read_write_retry_once_when_an_independent_writer_commits_new_generation(
    tmp_path, monkeypatch
):
    """A read discards its first snapshot when M2 proves a newer commit."""
    root = tmp_path / "read-write"
    reader = _store(root)
    writer = _store(root)
    key = "race-key"
    snapshots = 0
    authority_reads = 0
    try:
        reader.put("first", key=key)
        original_snapshot = reader.guarded_handler_io.open_snapshot
        original_read_entry = reader.lifecycle_authority.read_entry

        def count_reader_authority_reads(blob_key: str):
            nonlocal authority_reads
            authority_reads += 1
            return original_read_entry(blob_key)

        monkeypatch.setattr(
            reader.lifecycle_authority,
            "read_entry",
            count_reader_authority_reads,
        )

        @contextmanager
        def replace_before_first_snapshot(locator, metadata):
            nonlocal snapshots
            snapshots += 1
            with original_snapshot(locator, metadata) as snapshot:
                if snapshots == 1:
                    writer.put("second", key=key)
                yield snapshot

        reader.guarded_handler_io.open_snapshot = replace_before_first_snapshot

        assert reader.get(key) == "second"
        assert snapshots == 2
        assert authority_reads == 4
    finally:
        writer.close()
        reader.close()


def test_read_delete_race_retries_to_authoritative_absence(tmp_path):
    """A delete after M1 is observed as tombstoned absence, never a stale read."""
    root = tmp_path / "read-delete"
    reader = _store(root)
    deleter = _store(root)
    key = "race-key"
    snapshots = 0
    try:
        reader.put("payload", key=key)
        original_snapshot = reader.guarded_handler_io.open_snapshot

        @contextmanager
        def delete_before_first_snapshot(locator, metadata):
            nonlocal snapshots
            snapshots += 1
            if snapshots == 1:
                assert deleter.delete(key) is True
            with original_snapshot(locator, metadata) as snapshot:
                yield snapshot

        reader.guarded_handler_io.open_snapshot = delete_before_first_snapshot

        assert reader.get(key) is None
        assert snapshots == 1
    finally:
        deleter.close()
        reader.close()
