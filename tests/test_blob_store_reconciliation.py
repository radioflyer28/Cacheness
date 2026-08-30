"""Adversarial lifecycle-evidence contracts for BlobStore recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cacheness.error_handling import CacheManifestIntegrityError
from cacheness.storage import BlobStore
from cacheness.storage.integrity import sign_hmac_sha256, verify_hmac_sha256
from cacheness.storage.operation_record import (
    OPERATION_RECORD_SCHEMA_VERSION,
    LifecycleOperationRecord,
    OperationCheckpoint,
    OperationKind,
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
        "candidate_locator": str(root / "payload-generation-a"),
        "previous_locator": str(root / "payload-generation-b"),
        "transition": "replace",
        "checkpoint": OperationCheckpoint.PREPARED,
        "created_at": "2026-08-30T00:00:00+00:00",
        "updated_at": "2026-08-30T00:00:00+00:00",
    }
    values.update(overrides)
    return LifecycleOperationRecord(**values)  # type: ignore[arg-type]


def _signed_record(root: Path, key: bytes) -> LifecycleOperationRecord:
    record = _record(root)
    return record.with_signature(sign_hmac_sha256(record.signing_bytes(), key))


def test_operation_evidence_binds_provenance_transition_and_domain_signature(
    tmp_path: Path,
) -> None:
    """Recovery evidence has complete authenticated control provenance, not payload bytes."""
    key = b"operation-evidence-test-key-00001"
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
