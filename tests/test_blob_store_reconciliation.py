"""Authority-backed reconciliation contracts for BlobStore recovery."""

from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobRecoverableCleanupError
from cacheness.storage import BlobStore


def test_retired_scheduler_evidence_is_ignored_without_mutation(
    tmp_path: Path,
) -> None:
    """Unauthenticated predecessor evidence cannot influence authority recovery."""
    root = tmp_path / "retired-evidence"
    key = "authority-key"
    evidence_path = root / "operations" / ("a" * 32 + ".json")
    store = BlobStore(root, backend="json")
    try:
        assert store.put({"state": "committed"}, key=key) == key
        entry = store.lifecycle_authority.read_entry(key)
        assert entry is not None
        payload_path = root / entry.locator
        payload_before = payload_path.read_bytes()
        evidence_path.parent.mkdir()
        evidence = b'{"candidate_locator":"../outside","schema_version":2}'
        evidence_path.write_bytes(evidence)
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        report = reopened.reconcile()
        assert report.applied is False
        assert report.findings == ()
        assert evidence_path.read_bytes() == evidence
        assert payload_path.read_bytes() == payload_before
        assert reopened.get(key) == {"state": "committed"}
    finally:
        reopened.close()


def test_apply_reconciliation_reclaims_exact_debt_without_revoking_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only exact authenticated cleanup debt is replayed after a promotion."""
    store = BlobStore(tmp_path / "exact-debt", backend="json")
    try:
        key = store.put({"generation": "old"}, key="debt-key")
        previous = store.lifecycle_authority.read_entry(key)
        assert previous is not None
        old_payload = store.cache_dir / previous.locator
        original_cleanup = store._delete_or_prove_absent
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            lambda _locator: (_ for _ in ()).throw(OSError("defer cleanup")),
        )

        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.put({"generation": "winner"}, key=key)

        assert old_payload.exists()
        assert store.get(key) == {"generation": "winner"}

        monkeypatch.setattr(store, "_delete_or_prove_absent", original_cleanup)
        assert store.reconcile(apply=True).applied is True
        assert not old_payload.exists()
        assert store.get(key) == {"generation": "winner"}
    finally:
        store.close()
