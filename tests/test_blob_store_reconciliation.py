"""Adversarial lifecycle-evidence contracts for BlobStore recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cacheness.config import CacheConfig, LifecycleLimits
from cacheness.error_handling import (
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
        original_read = repository.file_ops.read_bytes

        def counting_read(locator: Path) -> bytes:
            calls.append(locator)
            return original_read(locator)

        monkeypatch.setattr(repository.file_ops, "read_bytes", counting_read)
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
