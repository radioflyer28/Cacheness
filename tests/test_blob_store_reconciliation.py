"""Adversarial lifecycle-evidence contracts for BlobStore recovery."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import base64
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread

import pytest
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from cacheness.config import CacheConfig, LifecycleLimits
from cacheness.metadata import InMemoryBackend
from cacheness.error_handling import (
    CacheBlobIntegrityError,
    CacheStorageError,
    CacheBlobLifecycleConflictError,
    CacheManifestIntegrityError,
)
from cacheness.storage import BlobStore
from cacheness.storage.reconciliation import ReconciliationAction, ReconciliationStatus
from cacheness.storage.integrity import (
    ManifestKeyProvider,
    sign_hmac_sha256,
    verify_hmac_sha256,
)
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
            "c" * 32,
            "a" * 32,
        ]
        assert first_page.next_cursor is not None
        # One bounded sequence head, one immutable event per yielded member,
        # and one exact control read per member; no history rewrite or
        # namespace-wide directory scan is used.
        assert len(calls) == 1 + (2 * limits.operation_page_size)
        assert [cursor.next_sequence for cursor in first_page.entry_next_cursors] == [2, 3]

        second_page = repository.list_page(first_page.next_cursor)
        assert [operation_id for operation_id, _ in second_page.entries] == ["b" * 32]
        assert second_page.next_cursor is None
        assert len(calls) == 8
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
        assert not list((root / "operations").glob("reconcile-action-*.json"))
    finally:
        store.close()


def test_reconciliation_candidate_cleanup_uses_durable_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconciliation cannot bypass the durable lifecycle cleanup primitive."""
    root = tmp_path / "reconcile-durable-delete"
    store = BlobStore(root, backend="json")
    try:
        key = store._manifest_key(initialize_new_store=True)
        record = _reconciliation_record(store, root, key, "a" * 32)
        candidate = store.guarded_handler_io.root / record.candidate_locator
        store.guarded_handler_io.file_ops.write_bytes_durable(candidate, b"candidate")
        store.lifecycle.operation_repository.create_exclusive(
            record, record.canonical_bytes()
        )
        file_ops = store.guarded_handler_io.file_ops
        original = file_ops.delete_durable
        calls: list[Path] = []

        def durable_delete(locator: Path) -> bool:
            calls.append(locator)
            return original(locator)

        monkeypatch.setattr(file_ops, "delete_durable", durable_delete)
        store.reconcile(
            apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc)
        )
        assert candidate in calls
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


def test_reopen_retires_an_orphaned_completed_reconciliation_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed action sidecar is exact-retired after a post-action crash."""
    root = tmp_path / "reconcile-orphaned-completed-checkpoint"
    store = BlobStore(root, backend="json")
    operation_id = "a" * 32
    try:
        key = store._manifest_key(initialize_new_store=True)
        record = _reconciliation_record(store, root, key, operation_id)
        candidate = store.guarded_handler_io.root / record.candidate_locator
        store.guarded_handler_io.file_ops.write_bytes_durable(candidate, b"candidate")
        store.lifecycle.operation_repository.create_exclusive(
            record, record.canonical_bytes()
        )
        original_retire = (
            store.lifecycle.operation_repository.retire_reconciliation_checkpoint_if_exact
        )

        def interrupt_checkpoint_retirement(*args: object, **kwargs: object) -> None:
            raise RuntimeError("interrupted after primary evidence retirement")

        monkeypatch.setattr(
            store.lifecycle.operation_repository,
            "retire_reconciliation_checkpoint_if_exact",
            interrupt_checkpoint_retirement,
        )
        with pytest.raises(RuntimeError, match="interrupted after primary"):
            store.reconcile(
                apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc)
            )
        assert not candidate.exists()
        assert store.lifecycle.operation_repository.get_raw(operation_id) is None
        assert list((root / "operations").glob("reconcile-action-*.json"))
        monkeypatch.setattr(
            store.lifecycle.operation_repository,
            "retire_reconciliation_checkpoint_if_exact",
            original_retire,
        )
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        reopened.reconcile(apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        assert not list((root / "operations").glob("reconcile-action-*.json"))
    finally:
        reopened.close()


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
def test_operation_leases_complete_after_releasing_parent_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_stripe: bool
) -> None:
    """Same/opposite stripes complete when child work never nests a parent lease."""
    store = BlobStore(tmp_path / "operation-scoped-clear-lease", backend="json")
    repository = store.lifecycle.operation_repository
    original_lock = operation_repository_module.interprocess_open_file_lock
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

    # Force an A→B / B→A schedule.  Each parent lease is released before its
    # child begins, so even one physical stripe cannot self-deadlock and two
    # stripes cannot form an opposite-order cycle.
    monkeypatch.setattr(operation_repository_module, "lock_stripe_index", forced_stripe)
    monkeypatch.setattr(
        operation_repository_module, "interprocess_open_file_lock", count_file_locks
    )
    first_entered = Event()
    release_first = Event()
    errors: list[BaseException] = []

    def run(operation_id: str, child_operation_id: str, entered: Event) -> None:
        try:
            with repository.operation_transition(operation_id):
                entered.set()
                if operation_id == "a" * 32:
                    assert release_first.wait(timeout=5)
            with repository.operation_transition(child_operation_id):
                pass
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
        # Each distinct record takes its own exact advisory authority lock.
        assert lock_calls == ["conditional_evidence"] * 4
    finally:
        release_first.set()
        store.close()


def test_operation_lease_reentrancy_is_exact_and_rejects_unordered_child(
    tmp_path: Path,
) -> None:
    """Only the same operation record may reuse an already-held authority lease."""
    store = BlobStore(tmp_path / "exact-operation-reentrancy", backend="json")
    repository = store.lifecycle.operation_repository
    parent = "a" * 32
    child = "b" * 32
    try:
        with repository.operation_transition(parent):
            with repository.operation_transition(parent):
                pass
            with pytest.raises(CacheBlobLifecycleConflictError, match="releasing the parent"):
                with repository.operation_transition(child):
                    pass
    finally:
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


def test_released_v1_reconciliation_token_vector_remains_resumable(tmp_path: Path) -> None:
    """The three-field token emitted before v2 decodes without sidecar state."""
    key = b"v1-reconciliation-token-key-0001"
    assert len(key) == 32
    root = tmp_path / "v1-token-vector"
    store = BlobStore(
        root,
        backend="json",
        manifest_key_provider=ManifestKeyProvider(root / "key.bin", key=key),
    )
    try:
        payload = json.dumps(
            {"manifest": "tenant/asset", "operation": "a" * 32, "priority": "operation"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        nonce = bytes(range(12))
        encrypted = ChaCha20Poly1305(store._reconciler._token_key()).encrypt(
            nonce, payload, store._reconciler._TOKEN_DOMAIN
        )
        vector = base64.urlsafe_b64encode(b"\x01" + nonce + encrypted).decode("ascii")
        assert vector == "AQABAgMEBQYHCAkKC6sC_5JPawWEAAeYChAEswIHp59Y0d-ZC8ycrixRmUeMdutGTss6Fcsw20KBPym3lBfPb3-sLf8yxzU8lQ3CnDaJRANQyq1k73Mo4LMCSYmsEm_A2iHxwJtgywpf5M3UBBa0-KImxny7-8MW4okhIZTG"
        assert store._reconciler._decode_resume_token(vector) == (
            reconciliation_module.ManifestCursor("tenant/asset"),
            reconciliation_module.OperationCursor("a" * 32),
            None,
            None,
            "operation",
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


def test_digest_invalid_pending_control_does_not_consume_recovery_action_budget(
    tmp_path: Path,
) -> None:
    """A syntax-valid bad digest stays untouched while a later valid control advances."""
    root = tmp_path / "pending-digest-budget"
    limits = _small_lifecycle_limits()
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=limits), backend="json"
    )
    try:
        repository = store.lifecycle.operation_repository
        operations = root / "operations"
        operations.mkdir(exist_ok=True)
        invalid_id = "0" * 32
        valid_id = "f" * 32
        invalid = operations / f".{invalid_id}.json.pending.{'0' * 64}.{'a' * 32}.tmp"
        invalid.write_bytes(b"digest-mismatch")
        valid_raw = b"valid-control"
        valid_digest = hashlib.sha256(valid_raw).hexdigest()
        valid = operations / f".{valid_id}.json.pending.{valid_digest}.{'b' * 32}.tmp"
        valid.write_bytes(valid_raw)

        assert repository.recover_pending_operation_records() == (valid_id,)
        assert invalid.read_bytes() == b"digest-mismatch"
        assert repository.get_raw(valid_id) == valid_raw
    finally:
        store.close()


def test_pending_recovery_pages_past_large_invalid_prefix_without_unbounded_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each recovery call reads one page and durable cursor progress reaches valid work."""
    root = tmp_path / "pending-cursor"
    limits = _small_lifecycle_limits()
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=limits), backend="json"
    )
    try:
        repository = store.lifecycle.operation_repository
        operations = root / "operations"
        operations.mkdir(exist_ok=True)
        for index in range(6):
            operation_id = f"{index:032x}"
            (operations / f".{operation_id}.json.pending.{'0' * 64}.{index:032x}.tmp").write_bytes(
                b"bad-digest"
            )
        valid_id = "f" * 32
        raw = b"valid-control"
        digest = hashlib.sha256(raw).hexdigest()
        (operations / f".{valid_id}.json.pending.{digest}.{'f' * 32}.tmp").write_bytes(raw)

        calls: list[Path] = []
        original_read = repository.file_ops.read_bytes_bounded

        def count_reads(locator: Path, *, max_bytes: int) -> bytes:
            calls.append(locator)
            return original_read(locator, max_bytes=max_bytes)

        monkeypatch.setattr(repository.file_ops, "read_bytes_bounded", count_reads)
        for iteration in range(4):
            before = len(calls)
            repository.recover_pending_operation_records()
            # The first legacy recovery performs one explicitly bounded
            # bootstrap into the high-water sequence. Later calls inspect only
            # the head, one immutable event window, and its exact candidates.
            assert len(calls) - before <= (
                limits.max_inventory_items + (2 * limits.operation_page_size) + 2
                if iteration == 0
                else 2 + (2 * limits.operation_page_size)
            )
        assert repository.get_raw(valid_id) == raw
        # The exact high-water sequence is terminal after the valid candidate;
        # retaining a lexical cursor here would replay the finished snapshot.
        assert not (operations / ".pending-recovery.cursor").exists()
    finally:
        store.close()


def test_dry_run_reports_blocked_pending_control_residue(tmp_path: Path) -> None:
    """A digest-invalid pending candidate is visible without becoming authority."""
    root = tmp_path / "pending-dry-run"
    store = BlobStore(root, backend="json")
    try:
        operations = root / "operations"
        operations.mkdir(exist_ok=True)
        operation_id = "a" * 32
        pending = operations / (
            f".{operation_id}.json.pending.{'0' * 64}.{'b' * 32}.tmp"
        )
        pending.write_bytes(b"wrong-digest")

        report = store.reconcile(now=datetime(2026, 8, 31, tzinfo=timezone.utc))

        assert any(
            finding.reason == "pending_control_untrusted" for finding in report.findings
        )
        assert pending.read_bytes() == b"wrong-digest"
        assert not (root / "blob_manifest_hmac_key.bin").exists()
    finally:
        store.close()


def test_pending_only_reconciliation_page_emits_and_resumes_its_cursor(
    tmp_path: Path,
) -> None:
    """Pending residue alone is truthful incomplete work, not a terminal page."""
    root = tmp_path / "pending-only-resume"
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=_small_lifecycle_limits()),
        backend="json",
    )
    try:
        store._manifest_key(initialize_new_store=True)
        operations = root / "operations"
        operations.mkdir(exist_ok=True)
        for index in range(3):
            operation_id = f"{index:032x}"
            (operations / (
                f".{operation_id}.json.pending.{'0' * 64}.{index:032x}.tmp"
            )).write_bytes(b"invalid")

        first = store.reconcile(now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        assert first.resume_token is not None
        decoded = store._reconciler._decode_resume_token(first.resume_token)
        assert decoded[3] is not None
        second = store.reconcile(
            resume_token=first.resume_token,
            now=datetime(2026, 8, 31, tzinfo=timezone.utc),
        )
        assert len(first.findings) == 2
        assert len(second.findings) == 1
        assert second.resume_token is None
    finally:
        store.close()


def test_invalid_checkpoint_does_not_starve_later_completed_orphan(
    tmp_path: Path,
) -> None:
    """Malformed sidecars remain blocked while eligible completed work converges."""
    from cacheness.storage.reconciliation import _ActionCheckpoint

    root = tmp_path / "checkpoint-budget"
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=_small_lifecycle_limits()),
        backend="json",
    )
    try:
        repository = store.lifecycle.operation_repository
        key = store._manifest_key(initialize_new_store=True)
        bad_id = "0" * 32
        good_id = "f" * 32
        repository.create_reconciliation_checkpoint_exclusive(bad_id, b"{")
        completed = _ActionCheckpoint.new(
            good_id,
            "a" * 64,
            ReconciliationAction.RETIRE_EVIDENCE,
            "completed",
            key,
        )
        raw = completed.canonical_bytes(lifecycle_limits=store.lifecycle_limits)
        repository.create_reconciliation_checkpoint_exclusive(good_id, raw)

        report = store.reconcile(
            apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc)
        )
        assert repository.get_reconciliation_checkpoint_raw(bad_id) == b"{"
        assert repository.get_reconciliation_checkpoint_raw(good_id) is None
        assert any(
            finding.reason == "reconciliation_checkpoint_untrusted"
            for finding in report.findings
        )
    finally:
        store.close()


def test_apply_charges_an_orphan_sidecar_once_before_later_primary_work(
    tmp_path: Path,
) -> None:
    """One completed orphan plus one safe primary exhaust a two-action budget exactly."""
    from cacheness.storage.reconciliation import _ActionCheckpoint

    root = tmp_path / "orphan-sidecar-one-charge"
    limits = LifecycleLimits(
        manifest_page_size=2,
        operation_page_size=2,
        max_reconcile_actions=2,
        orphan_grace_seconds=0.01,
        close_wait_seconds=0.02,
    )
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=limits), backend="json"
    )
    try:
        repository = store.lifecycle.operation_repository
        key = store._manifest_key(initialize_new_store=True)
        orphan_id = "a" * 32
        primary_id = "f" * 32
        completed = _ActionCheckpoint.new(
            orphan_id,
            "a" * 64,
            ReconciliationAction.RETIRE_EVIDENCE,
            "completed",
            key,
        )
        repository.create_reconciliation_checkpoint_exclusive(
            orphan_id,
            completed.canonical_bytes(lifecycle_limits=limits),
        )
        record = _reconciliation_record(store, root, key, primary_id)
        candidate = store.guarded_handler_io.root / record.candidate_locator
        store.guarded_handler_io.file_ops.write_bytes_durable(candidate, b"candidate")
        repository.create_exclusive(record, record.canonical_bytes(lifecycle_limits=limits))

        store.reconcile(apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc))

        assert repository.get_reconciliation_checkpoint_raw(orphan_id) is None
        assert repository.get_raw(primary_id) is None
        assert not candidate.exists()
    finally:
        store.close()


def test_signed_primary_mismatched_sidecar_blocks_only_its_primary(
    tmp_path: Path,
) -> None:
    """A signed foreign checkpoint cannot starve a later safe action."""
    from cacheness.storage.reconciliation import _ActionCheckpoint

    root = tmp_path / "mismatched-sidecar"
    store = BlobStore(
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=_small_lifecycle_limits()),
        backend="json",
    )
    try:
        key = store._manifest_key(initialize_new_store=True)
        repository = store.lifecycle.operation_repository
        blocked = _reconciliation_record(store, root, key, "a" * 32)
        safe = _reconciliation_record(store, root, key, "f" * 32)
        blocked_candidate = store.guarded_handler_io.root / blocked.candidate_locator
        safe_candidate = store.guarded_handler_io.root / safe.candidate_locator
        store.guarded_handler_io.file_ops.write_bytes_durable(blocked_candidate, b"blocked")
        store.guarded_handler_io.file_ops.write_bytes_durable(safe_candidate, b"safe")
        blocked_raw = blocked.canonical_bytes(lifecycle_limits=store.lifecycle_limits)
        repository.create_exclusive(blocked, blocked_raw)
        repository.create_exclusive(
            safe, safe.canonical_bytes(lifecycle_limits=store.lifecycle_limits)
        )
        foreign = _ActionCheckpoint.new(
            blocked.operation_id,
            "0" * 64,
            ReconciliationAction.DELETE_CANDIDATE,
            "prepared",
            key,
        )
        foreign_raw = foreign.canonical_bytes(lifecycle_limits=store.lifecycle_limits)
        repository.create_reconciliation_checkpoint_exclusive(
            blocked.operation_id, foreign_raw
        )

        report = store.reconcile(
            apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc)
        )

        assert any(
            finding.reason == "reconciliation_checkpoint_primary_mismatch"
            for finding in report.findings
        )
        assert repository.get_raw(blocked.operation_id) == blocked_raw
        assert repository.get_reconciliation_checkpoint_raw(blocked.operation_id) == foreign_raw
        assert not safe_candidate.exists()
        assert repository.get_raw(safe.operation_id) is None
    finally:
        store.close()


@pytest.mark.parametrize("backend", ("memory", "json", "sqlite"))
def test_pristine_reconcile_apply_is_an_idempotent_noop(tmp_path: Path, backend: str) -> None:
    """Apply does not create a manifest key merely to inspect an empty store."""
    root = tmp_path / f"pristine-{backend}"
    store = BlobStore(root, backend=InMemoryBackend() if backend == "memory" else backend)
    try:
        first = store.reconcile(apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        second = store.reconcile(apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        assert first.findings == second.findings == ()
        assert not (root / "blob_manifest_hmac_key.bin").exists()
    finally:
        store.close()


def test_dry_run_reports_malformed_matching_sidecar_without_deferring_startup(
    tmp_path: Path,
) -> None:
    """A pathname-only sidecar remains inert while valid tombstone recovery proceeds."""
    root = tmp_path / "matching-malformed-sidecar"
    store = BlobStore(root, backend="json")
    try:
        store.put({"value": "delete"}, key="tombstone-key")
        original_cleanup = store._delete_or_prove_absent
        store._delete_or_prove_absent = lambda _locator: (_ for _ in ()).throw(
            OSError("defer tombstone cleanup")
        )  # type: ignore[method-assign]
        with pytest.raises(CacheStorageError):
            store.delete("tombstone-key")
        store._delete_or_prove_absent = original_cleanup  # type: ignore[method-assign]
        operation_id, _raw = next(iter(store.lifecycle.operation_repository.list_page().entries))
        store.lifecycle.operation_repository.create_reconciliation_checkpoint_exclusive(
            operation_id, b"{"
        )
        report = store.reconcile(now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        assert any(
            finding.reason == "reconciliation_checkpoint_untrusted"
            for finding in report.findings
        )
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.manifest_repository.get_raw("tombstone-key") is None
    finally:
        reopened.close()


def test_tombstone_reconciliation_checkpoints_each_destructive_stage(
    tmp_path: Path,
) -> None:
    """A BaseException after deletion leaves primary evidence for a reopen resume."""
    class Interrupted(BaseException):
        pass

    root = tmp_path / "tombstone-checkpoint-stages"
    store = BlobStore(root, backend="json")
    try:
        store.put({"value": "delete"}, key="tombstone-key")
        original_cleanup = store._delete_or_prove_absent

        def leave_post_authority_debt(locator: Path) -> None:
            raise OSError("defer tombstone cleanup")

        store._delete_or_prove_absent = leave_post_authority_debt  # type: ignore[method-assign]
        with pytest.raises(CacheStorageError):
            store.delete("tombstone-key")
        store._delete_or_prove_absent = original_cleanup  # type: ignore[method-assign]

        def interrupt_after_payload(step: str, _record: LifecycleOperationRecord) -> None:
            if step == "reconcile_tombstone_after_payload_delete":
                raise Interrupted("after durable payload deletion")

        store.lifecycle.fault_hook = interrupt_after_payload
        with pytest.raises(Interrupted):
            store.reconcile(apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        operation_id, raw = next(iter(store.lifecycle.operation_repository.list_page().entries))
        assert raw
        checkpoint = store.lifecycle.operation_repository.get_reconciliation_checkpoint_raw(
            operation_id
        )
        assert checkpoint is not None
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        reopened.reconcile(apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        assert reopened.manifest_repository.get_raw("tombstone-key") is None
        assert reopened.lifecycle.operation_repository.list_page().entries == ()
        assert not list((root / "operations").glob("reconcile-action-*.json"))
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "seam",
    (
        "reconcile_tombstone_before_payload_delete",
        "reconcile_tombstone_inside_payload_delete",
        "reconcile_tombstone_after_payload_delete",
        "reconcile_tombstone_before_manifest_remove",
        "reconcile_tombstone_inside_manifest_remove",
        "reconcile_tombstone_after_manifest_remove",
        "reconcile_tombstone_before_payload_deleted_checkpoint",
        "reconcile_tombstone_inside_payload_deleted_checkpoint",
        "reconcile_tombstone_after_payload_deleted_checkpoint",
        "reconcile_tombstone_before_tombstone_removed_checkpoint",
        "reconcile_tombstone_inside_tombstone_removed_checkpoint",
        "reconcile_tombstone_after_tombstone_removed_checkpoint",
        "reconcile_tombstone_before_checkpoint_complete",
        "reconcile_tombstone_inside_checkpoint_complete",
        "reconcile_tombstone_after_checkpoint_complete",
        "reconcile_tombstone_before_primary_retire",
        "reconcile_tombstone_inside_primary_retire",
        "reconcile_tombstone_after_primary_retire",
        "reconcile_tombstone_before_sidecar_retire",
        "reconcile_tombstone_inside_sidecar_retire",
        "reconcile_tombstone_after_sidecar_retire",
    ),
)
def test_tombstone_fault_seams_converge_without_replaying_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seam: str,
) -> None:
    """Every lifecycle seam leaves one resumable destructive-effect history."""
    class Interrupted(BaseException):
        pass

    root = tmp_path / seam
    payload_deletes = 0
    payload_delete_calls = 0
    manifest_removals = 0
    payload_locator: Path | None = None

    def count_effects(active_store: BlobStore) -> None:
        nonlocal payload_deletes, payload_delete_calls, manifest_removals
        original_delete = active_store.guarded_handler_io.file_ops.delete_durable
        original_remove = active_store.manifest_repository.remove_if_expected

        def delete_once(locator: Path) -> bool:
            nonlocal payload_deletes, payload_delete_calls
            if locator == payload_locator:
                payload_delete_calls += 1
            deleted = original_delete(locator)
            if locator == payload_locator:
                payload_deletes += int(deleted)
            return deleted

        def remove_once(key: str, expectation: object) -> object:
            nonlocal manifest_removals
            manifest_removals += 1
            return original_remove(key, expectation)  # type: ignore[arg-type]

        monkeypatch.setattr(
            active_store.guarded_handler_io.file_ops, "delete_durable", delete_once
        )
        monkeypatch.setattr(active_store.manifest_repository, "remove_if_expected", remove_once)

    store = BlobStore(root, backend="json")
    try:
        store.put({"value": "delete"}, key="tombstone-key")
        original_cleanup = store._delete_or_prove_absent
        store._delete_or_prove_absent = lambda _locator: (_ for _ in ()).throw(
            OSError("defer tombstone cleanup")
        )  # type: ignore[method-assign]
        with pytest.raises(CacheStorageError):
            store.delete("tombstone-key")
        store._delete_or_prove_absent = original_cleanup  # type: ignore[method-assign]
        operation_id, raw = next(
            iter(store.lifecycle.operation_repository.list_page().entries)
        )
        recovered = store.lifecycle._recoverable_record(operation_id, raw)
        assert recovered is not None
        _record, payload_locator, _previous = recovered
        count_effects(store)

        def interrupt(step: str, _record: LifecycleOperationRecord) -> None:
            if step == seam:
                raise Interrupted(seam)

        store.lifecycle.fault_hook = interrupt
        with pytest.raises(Interrupted, match=seam):
            store.reconcile(apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc))
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        count_effects(reopened)
        reopened.reconcile(apply=True, now=datetime(2026, 8, 31, tzinfo=timezone.utc))
        assert payload_deletes == 1
        assert payload_delete_calls == 1
        assert manifest_removals == 1
        assert reopened.manifest_repository.get_raw("tombstone-key") is None
        assert reopened.lifecycle.operation_repository.list_page().entries == ()
        assert not list((root / "operations").glob("reconcile-action-*.json"))
    finally:
        reopened.close()
