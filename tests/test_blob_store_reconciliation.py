"""Adversarial lifecycle-evidence contracts for BlobStore recovery."""

from __future__ import annotations

import json
import multiprocessing
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread

import pytest

from cacheness.config import CacheConfig, LifecycleLimits
from cacheness.error_handling import (
    CacheBlobIntegrityError,
    CacheStorageError,
    CacheBlobLifecycleConflictError,
    CacheManifestIntegrityError,
)
from cacheness.storage import BlobStore
from cacheness.storage.reconciliation import ReconciliationAction, ReconciliationStatus
from cacheness.storage.integrity import sign_hmac_sha256, verify_hmac_sha256
from cacheness.storage.operation_record import (
    MAX_OPERATION_FIELD_BYTES,
    MAX_OPERATION_RECORD_BYTES,
    OPERATION_SIGNING_DOMAIN,
    OPERATION_RECORD_SCHEMA_VERSION,
    LifecycleOperationRecord,
    OperationCheckpoint,
    OperationKind,
    OperationTransition,
    store_identity,
)
from cacheness.storage.operation_repository import FileOperationRecordRepository
from cacheness.storage import operation_repository as operation_repository_module
from cacheness.storage.path_security import ManagedFileOps
from cacheness.storage import reconciliation as reconciliation_module


def _record(root: Path, **overrides: object) -> LifecycleOperationRecord:
    """Build one complete, unsigned record using the public evidence contract."""
    values: dict[str, object] = {
        "schema_version": OPERATION_RECORD_SCHEMA_VERSION,
        "operation_id": "a" * 32,
        "kind": OperationKind.PUT,
        "key": "tenant/asset",
        "owner": "cacheness.blob-store.lifecycle",
        "store_id": store_identity(str(root)),
        "topology": {"backend": "json", "root": store_identity(str(root))},
        "expected_generation": "b" * 32,
        "expected_record_digest": "c" * 64,
        "generation": "d" * 32,
        "candidate_locator": "payload-generation-a",
        "previous_locator": "payload-generation-b",
        "transition": OperationTransition.REPLACE,
        "checkpoint": OperationCheckpoint.PREPARED,
        "created_at": "2026-08-30T00:00:00+00:00",
        "updated_at": "2026-08-30T00:00:00+00:00",
    }
    values.update(overrides)
    return LifecycleOperationRecord(**values)  # type: ignore[arg-type]


def _signed_record(root: Path, key: bytes) -> LifecycleOperationRecord:
    record = _record(root)
    return record.with_signature(sign_hmac_sha256(record.signing_bytes(), key))


def _small_lifecycle_limits() -> LifecycleLimits:
    """Return deliberately small policy values for bounded lifecycle tests."""
    return LifecycleLimits(
        max_operation_record_bytes=1_048_576,
        max_operation_field_bytes=8_192,
        manifest_page_size=2,
        operation_page_size=2,
        max_reconcile_actions=1,
        orphan_grace_seconds=0.01,
        close_wait_seconds=0.02,
    )


def _record_for_operation(
    root: Path,
    key: bytes,
    operation_id: str,
) -> LifecycleOperationRecord:
    """Build signed, distinct evidence for one opaque repository entry."""
    record = _record(
        root,
        operation_id=operation_id,
        generation=operation_id,
        candidate_locator=f"candidate-{operation_id}",
        previous_locator=None,
        transition=OperationTransition.CREATE,
        expected_generation=None,
        expected_record_digest=None,
    )
    return record.with_signature(sign_hmac_sha256(record.signing_bytes(), key))


def _reconciliation_record(
    store: BlobStore,
    root: Path,
    key: bytes,
    operation_id: str,
) -> LifecycleOperationRecord:
    """Bind a synthetic record to the exact local store topology."""
    record = replace(
        _record_for_operation(root, key, operation_id),
        topology={
            "backend": type(store.backend).__name__,
            "root": store_identity(str(store.guarded_handler_io.root)),
        },
    )
    return record.with_signature(sign_hmac_sha256(record.signing_bytes(), key))


def _race_exact_evidence_transition_in_process(
    root: str,
    initial: bytes,
    updated: bytes,
    transition: str,
    ready: multiprocessing.queues.Queue,
    release: multiprocessing.synchronize.Event,
    outcomes: multiprocessing.queues.Queue,
) -> None:
    """Race a fresh managed root/repository without sharing interpreter locks."""
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(file_ops, lifecycle_limits=LifecycleLimits())
    try:
        # Pass only canonical evidence across the process boundary.  The
        # immutable record intentionally contains a MappingProxyType topology,
        # which is not a spawn-pickle transport; independently reopening the
        # exact signed record is also the production recovery model.
        record = LifecycleOperationRecord.from_canonical_bytes(initial)
        ready.put("ready")
        if not release.wait(timeout=10):
            outcomes.put("timeout")
            return
        if transition == "checkpoint":
            repository.checkpoint_if_exact(
                record.at_checkpoint(OperationCheckpoint.CANDIDATE_PUBLISHED),
                expected_raw=initial,
                raw_record=updated,
            )
        elif transition == "retire":
            repository.retire_if_exact(record, expected_raw=initial)
        else:  # pragma: no cover - test harness invariant.
            raise AssertionError(f"unknown transition: {transition}")
        outcomes.put("won")
    except CacheBlobLifecycleConflictError:
        outcomes.put("conflict")
    except BaseException as exc:  # pragma: no cover - surfaced by parent assertions.
        outcomes.put(f"error:{exc!r}")
    finally:
        file_ops.close()


def test_operation_evidence_binds_provenance_transition_and_domain_signature(
    tmp_path: Path,
) -> None:
    """Recovery evidence has complete authenticated control provenance, not payload bytes."""
    key = b"0123456789abcdef0123456789abcdef"
    record = _signed_record(tmp_path, key)

    assert LifecycleOperationRecord.from_canonical_bytes(record.canonical_bytes()) == record
    assert verify_hmac_sha256(record.signing_bytes(), record.signature, key)
    assert not verify_hmac_sha256(
        record.canonical_bytes(include_signature=False), record.signature, key
    )

    for checkpoint in (
        OperationCheckpoint.CANDIDATE_PUBLISHED,
        OperationCheckpoint.AUTHORITY_PUBLISHED,
        OperationCheckpoint.RECLAIMING,
        OperationCheckpoint.TERMINAL,
    ):
        record = record.at_checkpoint(checkpoint)

    with pytest.raises(CacheManifestIntegrityError, match="regressed"):
        record.at_checkpoint(OperationCheckpoint.PREPARED)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema_version":2,"schema_version":2}',
        b'{"schema_version":999}',
        b'{"schema_version":2,"owner":"attacker"}',
        b'{"schema_version":2,"candidate_locator":"../escape"}',
    ),
)
def test_untrusted_operation_evidence_rejects_ambiguous_control_bytes(raw: bytes) -> None:
    """Malformed evidence does not normalize into a lifecycle recovery record."""
    with pytest.raises(CacheManifestIntegrityError):
        LifecycleOperationRecord.from_canonical_bytes(raw)


def test_oversized_or_forged_evidence_is_preserved_without_recovery_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted evidence remains byte-identical and cannot trigger deletion on reopen."""
    root = tmp_path / "forged-operation"
    store = BlobStore(root, backend="json")
    try:
        key = store._manifest_key(initialize_new_store=True)
        raw = _signed_record(root, key).canonical_bytes()
        forged = json.loads(raw)
        forged["candidate_locator"] = "../outside"
        forged_raw = json.dumps(
            forged, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        evidence_path = root / "operations" / ("a" * 32 + ".json")
        evidence_path.parent.mkdir(exist_ok=True)
        evidence_path.write_bytes(forged_raw)
        before_mtime = evidence_path.stat().st_mtime_ns
        mutation_calls: list[Path] = []
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            lambda locator: mutation_calls.append(locator),
        )

        store.lifecycle.recover()

        assert evidence_path.read_bytes() == forged_raw
        assert evidence_path.stat().st_mtime_ns == before_mtime
        assert mutation_calls == []
    finally:
        store.close()


def test_operation_record_field_and_raw_byte_bounds_are_checked_before_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control metadata rejects one byte beyond each explicit evidence boundary."""
    accepted = _record(tmp_path, key="k" * MAX_OPERATION_FIELD_BYTES)
    assert accepted.key == "k" * MAX_OPERATION_FIELD_BYTES
    with pytest.raises(CacheManifestIntegrityError, match="key exceeds"):
        _record(tmp_path, key="k" * (MAX_OPERATION_FIELD_BYTES + 1))

    parser_called = False

    def reject_parser(*_args: object, **_kwargs: object) -> object:
        nonlocal parser_called
        parser_called = True
        raise AssertionError("oversized evidence must not reach JSON parsing")

    import cacheness.storage.operation_record as operation_record

    monkeypatch.setattr(operation_record.json, "loads", reject_parser)
    with pytest.raises(CacheManifestIntegrityError, match="byte limit"):
        LifecycleOperationRecord.from_canonical_bytes(b"x" * (MAX_OPERATION_RECORD_BYTES + 1))
    assert not parser_called


def test_operation_repository_uses_configured_bounded_stable_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small caller policy reaches opaque pages without over-reading entries."""
    root = tmp_path / "paged-operation-records"
    limits = _small_lifecycle_limits()
    config = CacheConfig(cache_dir=str(root), lifecycle_limits=limits)
    store = BlobStore(config=config, backend="json")
    try:
        key = store._manifest_key(initialize_new_store=True)
        repository = store.lifecycle.operation_repository
        assert repository.lifecycle_limits is limits

        for operation_id in ("c" * 32, "a" * 32, "b" * 32):
            record = _record_for_operation(root, key, operation_id)
            repository.create_exclusive(record, record.canonical_bytes())

        calls: list[Path] = []
        original_read = repository.file_ops.read_bytes_bounded

        def counting_read(locator: Path, *, max_bytes: int) -> bytes:
            calls.append(locator)
            return original_read(locator, max_bytes=max_bytes)

        monkeypatch.setattr(repository.file_ops, "read_bytes_bounded", counting_read)
        first_page = repository.list_page()

        assert [operation_id for operation_id, _ in first_page.entries] == [
            "a" * 32,
            "b" * 32,
        ]
        assert first_page.next_cursor is not None
        assert len(calls) == limits.operation_page_size

        second_page = repository.list_page(first_page.next_cursor)
        assert [operation_id for operation_id, _ in second_page.entries] == ["c" * 32]
        assert second_page.next_cursor is None
        assert len(calls) == 3
    finally:
        store.close()


def test_operation_checkpoint_and_retirement_require_exact_prior_bytes(
    tmp_path: Path,
) -> None:
    """Stale recovery work cannot overwrite or retire newer operation evidence."""
    root = tmp_path / "exact-operation-records"
    store = BlobStore(root, backend="json")
    try:
        key = store._manifest_key(initialize_new_store=True)
        repository = store.lifecycle.operation_repository
        original = _record_for_operation(root, key, "a" * 32)
        original_raw = original.canonical_bytes()
        repository.create_exclusive(original, original_raw)
        updated = original.at_checkpoint(
            OperationCheckpoint.CANDIDATE_PUBLISHED,
            updated_at="2026-08-30T00:00:01+00:00",
        )
        updated = updated.with_signature(sign_hmac_sha256(updated.signing_bytes(), key))
        updated_raw = updated.canonical_bytes()

        repository.checkpoint_if_exact(
            updated, expected_raw=original_raw, raw_record=updated_raw
        )
        with pytest.raises(CacheBlobLifecycleConflictError):
            repository.retire_if_exact(original, expected_raw=original_raw)
        assert repository.get_raw(original.operation_id) == updated_raw

        repository.retire_if_exact(updated, expected_raw=updated_raw)
        assert repository.get_raw(updated.operation_id) is None
    finally:
        store.close()


def test_configured_orphan_grace_delays_authenticated_candidate_eligibility(
    tmp_path: Path,
) -> None:
    """Age is only one input after authenticated evidence revalidation."""
    root = tmp_path / "orphan-grace"
    limits = _small_lifecycle_limits()
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=limits), backend="json"
    )
    try:
        key = store._manifest_key(initialize_new_store=True)
        record = _record_for_operation(root, key, "a" * 32).at_checkpoint(
            OperationCheckpoint.CANDIDATE_PUBLISHED,
            updated_at="2026-08-30T00:00:00+00:00",
        )
        record = record.with_signature(sign_hmac_sha256(record.signing_bytes(), key))
        created = datetime(2026, 8, 30, tzinfo=timezone.utc)

        assert not store.lifecycle.is_pre_authority_candidate_eligible(
            record, now=created + timedelta(seconds=0.009)
        )
        assert store.lifecycle.is_pre_authority_candidate_eligible(
            record, now=created + timedelta(seconds=0.01)
        )
    finally:
        store.close()


@pytest.mark.parametrize("defect", ("wrong_owner", "wrong_store", "wrong_operation", "escape"))
def test_invalid_evidence_never_reaches_a_recovery_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    """Every provenance failure is retained exactly and remains non-destructive."""
    root = tmp_path / defect
    store = BlobStore(root, backend="json")
    try:
        key = store._manifest_key(initialize_new_store=True)
        record = _signed_record(root, key)
        mapping = record.to_mapping()
        filename = record.operation_id
        if defect == "wrong_owner":
            mapping["owner"] = "attacker"
        elif defect == "wrong_store":
            foreign_store = store_identity("foreign-root")
            mapping["store_id"] = foreign_store
            mapping["topology"] = {"backend": "JsonBackend", "root": foreign_store}
        elif defect == "wrong_operation":
            filename = "b" * 32
        else:
            mapping["candidate_locator"] = "../outside"
        unsigned = dict(mapping)
        unsigned.pop("signature", None)
        signing_bytes = OPERATION_SIGNING_DOMAIN + json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        mapping["signature"] = sign_hmac_sha256(signing_bytes, key)
        raw = json.dumps(
            mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        evidence_path = root / "operations" / f"{filename}.json"
        evidence_path.parent.mkdir(exist_ok=True)
        evidence_path.write_bytes(raw)
        before_mtime = evidence_path.stat().st_mtime_ns
        mutation_calls: list[Path] = []
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            lambda locator: mutation_calls.append(locator),
        )

        store.lifecycle.recover()

        assert evidence_path.read_bytes() == raw
        assert evidence_path.stat().st_mtime_ns == before_mtime
        assert mutation_calls == []
    finally:
        store.close()


def test_reconcile_dry_run_is_deterministic_and_does_not_mutate_candidate_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run reports authenticated, aged candidate debt without repairing it."""
    root = tmp_path / "reconcile-dry-run"
    limits = _small_lifecycle_limits()
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=limits), backend="json"
    )
    try:
        key = store._manifest_key(initialize_new_store=True)
        record = _reconciliation_record(store, root, key, "a" * 32)
        candidate = store.guarded_handler_io.root / record.candidate_locator
        store.guarded_handler_io.file_ops.write_bytes_durable(candidate, b"candidate")
        store.lifecycle.operation_repository.create_exclusive(
            record, record.canonical_bytes()
        )
        before = (candidate.read_bytes(), store.lifecycle.operation_repository.get_raw(record.operation_id))
        handler_calls: list[str] = []
        monkeypatch.setattr(
            store.handlers,
            "get_handler",
            lambda _data: handler_calls.append("handler"),
        )

        first = store.reconcile(now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        second = store.reconcile(now=datetime(2026, 8, 31, tzinfo=timezone.utc))

        assert first.to_dict() == second.to_dict()
        assert first.human_summary == second.human_summary
        assert first.findings[0].status is ReconciliationStatus.SAFE
        assert first.findings[0].action is ReconciliationAction.DELETE_CANDIDATE
        assert "tenant/asset" not in repr(first.to_dict())
        assert before == (
            candidate.read_bytes(),
            store.lifecycle.operation_repository.get_raw(record.operation_id),
        )
        assert handler_calls == []
        assert store._reconciler.lifecycle_limits is limits
    finally:
        store.close()


def test_reconcile_dry_run_is_bounded_and_exposes_an_opaque_resume_token(
    tmp_path: Path,
) -> None:
    """A short action budget reports deterministic partial progress without writes."""
    root = tmp_path / "reconcile-bounded"
    limits = _small_lifecycle_limits()
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=limits), backend="json"
    )
    try:
        key = store._manifest_key(initialize_new_store=True)
        for operation_id in ("a" * 32, "b" * 32):
            record = _reconciliation_record(store, root, key, operation_id)
            store.lifecycle.operation_repository.create_exclusive(
                record, record.canonical_bytes()
            )

        report = store.reconcile(now=datetime(2026, 8, 31, tzinfo=timezone.utc))

        assert len(report.findings) == limits.max_reconcile_actions
        assert report.resume_token is not None
        assert "a" * 32 not in report.resume_token
        assert report.to_dict()["resume_token"] == report.resume_token
    finally:
        store.close()


def test_reconcile_apply_revalidates_and_removes_only_authenticated_candidate(
    tmp_path: Path,
) -> None:
    """Apply rechecks exact evidence before reclaiming an aged owned candidate."""
    root = tmp_path / "reconcile-apply"
    store = BlobStore(root, backend="json")
    try:
        key = store._manifest_key(initialize_new_store=True)
        record = _reconciliation_record(store, root, key, "a" * 32)
        candidate = store.guarded_handler_io.root / record.candidate_locator
        store.guarded_handler_io.file_ops.write_bytes_durable(candidate, b"candidate")
        store.lifecycle.operation_repository.create_exclusive(
            record, record.canonical_bytes()
        )

        report = store.reconcile(
            apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc)
        )

        assert report.applied
        assert report.findings[0].action is ReconciliationAction.DELETE_CANDIDATE
        assert not candidate.exists()
        assert store.lifecycle.operation_repository.get_raw(record.operation_id) is None
    finally:
        store.close()


def test_reconcile_apply_base_exception_checkpoints_without_repeating_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-delete interruption leaves a completed checkpoint before re-raising."""
    class Interrupted(BaseException):
        pass

    root = tmp_path / "reconcile-base-exception"
    store = BlobStore(root, backend="json")
    try:
        key = store._manifest_key(initialize_new_store=True)
        record = _reconciliation_record(store, root, key, "a" * 32)
        candidate = store.guarded_handler_io.root / record.candidate_locator
        store.guarded_handler_io.file_ops.write_bytes_durable(candidate, b"candidate")
        store.lifecycle.operation_repository.create_exclusive(
            record, record.canonical_bytes()
        )
        delete = store.guarded_handler_io.file_ops.delete

        def delete_then_interrupt(locator: Path) -> bool:
            assert delete(locator)
            raise Interrupted("after candidate deletion")

        monkeypatch.setattr(store.guarded_handler_io.file_ops, "delete", delete_then_interrupt)
        with pytest.raises(Interrupted):
            store.reconcile(
                apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc)
            )
        assert not candidate.exists()
        assert store.lifecycle.operation_repository.get_raw(record.operation_id) is None
    finally:
        store.close()


def test_reconcile_apply_resumes_bounded_actions_from_opaque_token(tmp_path: Path) -> None:
    """A second apply consumes only the remaining evidence after a cutoff."""
    root = tmp_path / "reconcile-resume"
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=_small_lifecycle_limits()),
        backend="json",
    )
    try:
        key = store._manifest_key(initialize_new_store=True)
        candidates: list[Path] = []
        for operation_id in ("a" * 32, "b" * 32):
            record = _reconciliation_record(store, root, key, operation_id)
            candidate = store.guarded_handler_io.root / record.candidate_locator
            store.guarded_handler_io.file_ops.write_bytes_durable(candidate, operation_id.encode())
            store.lifecycle.operation_repository.create_exclusive(
                record, record.canonical_bytes()
            )
            candidates.append(candidate)

        first = store.reconcile(
            apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc)
        )
        assert first.resume_token is not None
        assert not candidates[0].exists()
        assert candidates[1].exists()

        second = store.reconcile(
            apply=True,
            resume_token=first.resume_token,
            now=datetime(2026, 8, 31, tzinfo=timezone.utc),
        )
        assert second.resume_token is None
        assert not candidates[1].exists()
    finally:
        store.close()


def test_reconcile_resume_interleaves_manifest_and_operation_pages(tmp_path: Path) -> None:
    """An authenticated manifest page cannot starve a later operation page."""
    root = tmp_path / "reconcile-interleave"
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=_small_lifecycle_limits()),
        backend="json",
    )
    try:
        store.put("committed", key="live-manifest")
        key = store._manifest_key()
        record = _reconciliation_record(store, root, key, "a" * 32)
        candidate = store.guarded_handler_io.root / record.candidate_locator
        store.guarded_handler_io.file_ops.write_bytes_durable(candidate, b"candidate")
        store.lifecycle.operation_repository.create_exclusive(
            record, record.canonical_bytes()
        )

        first = store.reconcile(now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        assert first.resume_token is not None
        second = store.reconcile(
            resume_token=first.resume_token,
            now=datetime(2026, 8, 31, tzinfo=timezone.utc),
        )

        assert second.findings[0].action is ReconciliationAction.DELETE_CANDIDATE
        assert candidate.exists()
    finally:
        store.close()


@pytest.mark.parametrize(
    ("transitions", "expected_terminal"),
    (
        (("checkpoint", "checkpoint"), "checkpoint"),
        (("checkpoint", "retire"), "either"),
    ),
)
def test_independent_processes_have_one_exact_evidence_transition_winner(
    tmp_path: Path,
    transitions: tuple[str, str],
    expected_terminal: str,
) -> None:
    """Checkpoint/CAS races cross real process and descriptor boundaries."""
    root = tmp_path / "operation-cas-independent"
    store = BlobStore(root, backend="json")
    context = multiprocessing.get_context("spawn")
    workers: list[multiprocessing.Process] = []
    try:
        key = store._manifest_key(initialize_new_store=True)
        record = _reconciliation_record(store, root, key, "a" * 32)
        initial = record.canonical_bytes(lifecycle_limits=store.lifecycle_limits)
        updated = record.at_checkpoint(
            OperationCheckpoint.CANDIDATE_PUBLISHED
        ).canonical_bytes(lifecycle_limits=store.lifecycle_limits)
        repository = FileOperationRecordRepository(
            store.guarded_handler_io.file_ops, lifecycle_limits=store.lifecycle_limits
        )
        repository.create_exclusive(record, initial)

        ready = context.Queue()
        release = context.Event()
        outcomes = context.Queue()
        for transition in transitions:
            worker = context.Process(
                target=_race_exact_evidence_transition_in_process,
                args=(
                    str(root),
                    initial,
                    updated,
                    transition,
                    ready,
                    release,
                    outcomes,
                ),
            )
            worker.start()
            workers.append(worker)
        assert ready.get(timeout=10) == "ready"
        assert ready.get(timeout=10) == "ready"
        release.set()
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0

        observed = sorted(outcomes.get(timeout=10) for _ in workers)
        assert observed == ["conflict", "won"]
        current = repository.get_raw(record.operation_id)
        if expected_terminal == "checkpoint":
            assert current == updated
        else:
            assert current in {None, updated}
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        store.close()


def test_evidence_transition_lock_files_are_fixed_bounded_stripes(tmp_path: Path) -> None:
    """High-cardinality successful transitions retain only the fixed stripe set."""
    root = tmp_path / "bounded-evidence-locks"
    store = BlobStore(root, backend="json")
    try:
        key = store._manifest_key(initialize_new_store=True)
        repository = store.lifecycle.operation_repository
        for index in range(160):
            operation_id = f"{index:032x}"
            record = _reconciliation_record(store, root, key, operation_id)
            initial = record.canonical_bytes(lifecycle_limits=store.lifecycle_limits)
            updated_record = record.at_checkpoint(OperationCheckpoint.CANDIDATE_PUBLISHED)
            updated = updated_record.canonical_bytes(
                lifecycle_limits=store.lifecycle_limits
            )
            repository.create_exclusive(record, initial)
            repository.checkpoint_if_exact(
                updated_record, expected_raw=initial, raw_record=updated
            )
            repository.retire_if_exact(updated_record, expected_raw=updated)

        lock_files = list((root / "operations" / ".conditional-locks").glob("*.lock"))
        assert 0 < len(lock_files) <= 64
        assert all(path.name[:-5].isalnum() and len(path.name) == 7 for path in lock_files)
    finally:
        store.close()


@pytest.mark.parametrize("same_stripe", (True, False))
def test_clear_operation_lease_avoids_nested_same_stripe_and_opposite_order_deadlocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_stripe: bool
) -> None:
    """An outer clear lease prevents same-stripe and A/B child lock cycles."""
    store = BlobStore(tmp_path / "operation-scoped-clear-lease", backend="json")
    repository = store.lifecycle.operation_repository
    original_lock = operation_repository_module.interprocess_file_lock
    lock_calls: list[str] = []

    @contextmanager
    def count_file_locks(*args, **kwargs):
        lock_calls.append(kwargs["operation"])
        with original_lock(*args, **kwargs):
            yield

    def forced_stripe(_root: Path, identity: str) -> int:
        if same_stripe:
            return 0
        return 0 if identity.endswith("a" * 32) else 1

    # Force an outer A→child B / outer B→child A topology. On one stripe it
    # catches self-deadlock through a second descriptor; on two stripes it
    # catches the classic opposite-order cycle.
    monkeypatch.setattr(operation_repository_module, "lock_stripe_index", forced_stripe)
    monkeypatch.setattr(operation_repository_module, "interprocess_file_lock", count_file_locks)
    first_entered = Event()
    release_first = Event()
    errors: list[BaseException] = []

    def run(operation_id: str, child_operation_id: str, entered: Event) -> None:
        try:
            with repository.operation_transition(operation_id):
                with repository.operation_transition(child_operation_id):
                    entered.set()
                    if operation_id == "a" * 32:
                        assert release_first.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - surfaced below.
            errors.append(exc)

    first = Thread(
        target=run,
        args=("a" * 32, "b" * 32, first_entered),
    )
    second_entered = Event()
    second = Thread(
        target=run,
        args=("b" * 32, "a" * 32, second_entered),
    )
    try:
        first.start()
        assert first_entered.wait(timeout=5)
        second.start()
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
        assert not first.is_alive()
        assert not second.is_alive()
        assert second_entered.is_set()
        assert errors == []
        # One outer lease each; neither nested child operation opens a file lock.
        assert lock_calls == ["conditional_evidence", "conditional_evidence"]
    finally:
        release_first.set()
        store.close()


def test_evidence_limit_is_enforced_before_any_unbounded_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured instance limit rejects raw evidence before full reads."""
    root = tmp_path / "bounded-evidence"
    limits = LifecycleLimits(
        max_operation_record_bytes=64,
        max_operation_field_bytes=32,
        manifest_page_size=1,
        operation_page_size=1,
        max_reconcile_actions=1,
        orphan_grace_seconds=1,
        close_wait_seconds=1,
    )
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=limits), backend="json"
    )
    try:
        repository = store.lifecycle.operation_repository
        operation_id = "a" * 32
        locator = repository.locator_for(operation_id)
        repository.file_ops.write_bytes_durable(locator, b"x" * 65)
        monkeypatch.setattr(
            repository.file_ops,
            "read_bytes",
            lambda _locator: (_ for _ in ()).throw(
                AssertionError("must not make an unbounded evidence read")
            ),
        )

        with pytest.raises(CacheManifestIntegrityError):
            repository.get_raw(operation_id)
    finally:
        store.close()


def test_reconciliation_tokens_are_nonce_random_authenticated_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume cursors are opaque AEAD values and reject abuse before decoding."""
    root = tmp_path / "opaque-resume-token"
    store = BlobStore(
        config=CacheConfig(
            cache_dir=str(root), lifecycle_limits=_small_lifecycle_limits()
        ),
        backend="json",
    )
    try:
        key = store._manifest_key(initialize_new_store=True)
        for operation_id in ("a" * 32, "b" * 32):
            record = _reconciliation_record(store, root, key, operation_id)
            store.lifecycle.operation_repository.create_exclusive(
                record, record.canonical_bytes(lifecycle_limits=store.lifecycle_limits)
            )

        first = store.reconcile(now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        second = store.reconcile(now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        assert first.resume_token is not None
        assert second.resume_token is not None
        assert first.resume_token != second.resume_token
        assert store._reconciler._decode_resume_token(first.resume_token) == (
            store._reconciler._decode_resume_token(second.resume_token)
        )
        tampered = first.resume_token[:-1] + (
            "A" if first.resume_token[-1] != "A" else "B"
        )
        with pytest.raises(ValueError, match="invalid"):
            store._reconciler._decode_resume_token(tampered)

        monkeypatch.setattr(
            reconciliation_module.base64,
            "urlsafe_b64decode",
            lambda _token: (_ for _ in ()).throw(AssertionError("must not decode")),
        )
        with pytest.raises(ValueError, match="byte limit"):
            store._reconciler._decode_resume_token(
                "x" * (store.lifecycle_limits.max_operation_field_bytes * 4)
            )
    finally:
        store.close()


def test_reconciliation_public_types_and_reason_coded_errors_are_narrow() -> None:
    """Storage exports report values and exact reconciliation error boundaries."""
    from cacheness.storage import (
        CacheBlobReconciliationCheckpointError,
        CacheBlobReconciliationConflictError,
        CacheBlobReconciliationError,
        ReconciliationAction,
        ReconciliationFinding,
        ReconciliationReport,
        ReconciliationStatus,
    )

    finding = ReconciliationFinding(
        status=ReconciliationStatus.BLOCKED,
        action=ReconciliationAction.REPORT_ONLY,
        reason="manifest_untrusted",
    )
    report = ReconciliationReport(
        findings=(finding,),
        resume_token=None,
        applied=False,
        manifest_records_seen=1,
        operation_records_seen=0,
    )
    assert report.to_dict()["findings"][0]["reason"] == "manifest_untrusted"
    assert issubclass(CacheBlobReconciliationError, CacheStorageError)
    assert issubclass(CacheBlobReconciliationConflictError, CacheStorageError)
    assert issubclass(CacheBlobReconciliationCheckpointError, CacheBlobIntegrityError)
    assert CacheBlobReconciliationError("blocked").context["reason"] == (
        "blob_reconciliation_blocked"
    )
