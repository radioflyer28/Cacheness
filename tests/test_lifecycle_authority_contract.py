"""Wave 0 compatibility and deterministic lifecycle-authority contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from _lifecycle_test_support import (
    BoundaryHooks,
    authority_root_snapshot,
    classify_authority_evidence,
)
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobCloseTimeoutError,
    CacheBlobIntegrityError,
    CacheBlobLifecycleConflictError,
    CacheBlobReconciliationCheckpointError,
    CacheBlobReconciliationConflictError,
    CacheBlobReconciliationError,
    CacheBlobRecoverableCleanupError,
    CacheBlobStoreClosedError,
    CacheBlobLifecycleTimeoutError,
    CacheManifestIntegrityError,
    CacheReason,
    CacheStorageError,
)
from cacheness.storage.reconciliation import (
    ReconciliationAction,
    ReconciliationFinding,
    ReconciliationReport,
    ReconciliationStatus,
)


def test_lifecycle_reason_values_and_typed_error_bases_are_frozen() -> None:
    """Authority work cannot rename the direct BlobStore failure contract."""
    expected_reasons = {
        "BLOB_LIFECYCLE_CONFLICT": "blob_lifecycle_conflict",
        "BLOB_BACKEND_FAILURE": "blob_backend_failure",
        "BLOB_BACKEND_CAPABILITY_UNSUPPORTED": "blob_backend_capability_unsupported",
        "BLOB_MIGRATION_REQUIRED": "blob_migration_required",
        "BLOB_RECOVERABLE_CLEANUP": "blob_recoverable_cleanup",
        "BLOB_RECONCILIATION_BLOCKED": "blob_reconciliation_blocked",
        "BLOB_RECONCILIATION_CONFLICT": "blob_reconciliation_conflict",
        "BLOB_RECONCILIATION_CHECKPOINT_INVALID": "blob_reconciliation_checkpoint_invalid",
        "BLOB_STORE_CLOSED": "blob_store_closed",
        "BLOB_CLOSE_TIMEOUT": "blob_close_timeout",
        "BLOB_LIFECYCLE_TIMEOUT": "blob_lifecycle_timeout",
    }

    assert {name: CacheReason[name].value for name in expected_reasons} == expected_reasons
    assert issubclass(CacheBlobIntegrityError, CacheStorageError)
    assert issubclass(CacheManifestIntegrityError, CacheBlobIntegrityError)
    assert issubclass(CacheBlobLifecycleConflictError, CacheStorageError)
    assert issubclass(CacheBlobBackendError, CacheStorageError)
    assert issubclass(CacheBlobRecoverableCleanupError, CacheStorageError)
    assert issubclass(CacheBlobReconciliationError, CacheStorageError)
    assert issubclass(CacheBlobReconciliationConflictError, CacheBlobReconciliationError)
    assert issubclass(CacheBlobReconciliationCheckpointError, CacheBlobIntegrityError)
    assert issubclass(CacheBlobStoreClosedError, CacheStorageError)
    assert issubclass(CacheBlobCloseTimeoutError, CacheStorageError)
    assert issubclass(CacheBlobLifecycleTimeoutError, CacheStorageError)


def test_reconciliation_v1_dictionary_and_human_summary_are_exact() -> None:
    """The zero-argument dictionaries stay the legacy v1 compatibility view."""
    finding = ReconciliationFinding(
        status=ReconciliationStatus.SAFE,
        action=ReconciliationAction.DELETE_CANDIDATE,
        reason="candidate_unpublished",
        evidence_id="operation-1",
        evidence_digest="evidence-digest",
        manifest_digest="manifest-digest",
        key_fingerprint="key-fingerprint",
        locator_fingerprint="locator-fingerprint",
    )
    report = ReconciliationReport(
        findings=(finding,),
        resume_token="resume-1",
        applied=False,
        manifest_records_seen=2,
        operation_records_seen=3,
    )

    assert finding.to_dict() == {
        "status": "safe",
        "action": "delete_candidate",
        "reason": "candidate_unpublished",
        "evidence_id": "operation-1",
        "evidence_digest": "evidence-digest",
        "manifest_digest": "manifest-digest",
        "key_fingerprint": "key-fingerprint",
        "locator_fingerprint": "locator-fingerprint",
    }
    assert report.to_dict() == {
        "applied": False,
        "findings": [finding.to_dict()],
        "manifest_records_seen": 2,
        "operation_records_seen": 3,
        "resume_token": "resume-1",
        "human_summary": (
            "Reconciliation (dry-run): 1 finding(s); safe=1, blocked=0, "
            "requires_confirmation=0, resume=yes"
        ),
    }
    assert report.human_summary == report.to_dict()["human_summary"]


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        (None, "empty"),
        ("payload.bin", "payload_without_authority"),
        ("cache_metadata.json", "metadata_without_authority"),
        ("provenance.json", "legacy_without_authority"),
        (".cacheness-inventory-v2", "scheduler_without_authority"),
        ("lifecycle-authority-v2.sqlite3", "future_authority"),
        ("lifecycle-authority-v1.sqlite3", "corrupt_authority"),
        ("mixed", "mixed_without_authority"),
    ],
)
def test_established_authority_missing_evidence_is_not_empty(
    tmp_path: Path, artifact: str | None, expected: str
) -> None:
    """Classification is read-only and keeps every migration/rebuild case distinct."""
    root = tmp_path / "store"
    if artifact is not None:
        root.mkdir()
        if artifact == "mixed":
            (root / "payload.bin").write_bytes(b"payload")
            (root / "cache_metadata.json").write_text("{}", encoding="utf-8")
        elif artifact == ".cacheness-inventory-v2":
            (root / artifact).mkdir()
        else:
            (root / artifact).write_bytes(b"unrecognized")

    before = authority_root_snapshot(root)
    assert classify_authority_evidence(root) == expected
    assert authority_root_snapshot(root) == before


def test_wrong_object_and_deterministic_boundary_observers_are_explicit(
    tmp_path: Path,
) -> None:
    """Ordering evidence uses hooks, never sleeps or timing races."""
    wrong_root = tmp_path / "not-a-directory"
    wrong_root.write_bytes(b"not a store")
    before = authority_root_snapshot(wrong_root)

    assert classify_authority_evidence(wrong_root) == "wrong_root_object"
    assert authority_root_snapshot(wrong_root) == before

    hooks = BoundaryHooks()
    observed: list[str] = []
    hooks.add_observer(observed.append)
    for boundary in (
        "authority.transaction.begin",
        "payload.stage",
        "payload.publish",
        "payload.verify",
        "payload.cleanup",
        "authority.transaction.end",
    ):
        hooks.reach(boundary)

    assert observed == [
        "authority.transaction.begin",
        "payload.stage",
        "payload.publish",
        "payload.verify",
        "payload.cleanup",
        "authority.transaction.end",
    ]
