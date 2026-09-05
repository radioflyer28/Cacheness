"""Authority-backed reconciliation contracts for BlobStore recovery."""

from pathlib import Path

import pytest

from cacheness import CacheConfig
from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobMigrationRequiredError,
    CacheBlobRecoverableCleanupError,
)
from cacheness.storage import BlobStore


def test_retired_control_requires_rebuild_without_mutation(
    tmp_path: Path,
) -> None:
    """A predecessor control cannot be replayed into an authority store."""
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

    with pytest.raises(CacheBlobMigrationRequiredError):
        BlobStore(root, backend="json")

    assert evidence_path.read_bytes() == evidence
    assert payload_path.read_bytes() == payload_before
    assert not (root / ".cacheness" / "lifecycle-authority-v1.sqlite3-journal").exists()


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


def test_authority_reconciliation_exposes_stable_v2_machine_view_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run reports one redacted authority finding without changing state."""
    store = BlobStore(tmp_path / "v2-authority-report", backend="json")
    try:
        key = store.put({"generation": "old"}, key="report-key")
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            lambda _locator: (_ for _ in ()).throw(OSError("defer cleanup")),
        )
        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.put({"generation": "winner"}, key=key)

        before = store.lifecycle_authority.snapshot_state()
        monkeypatch.setattr(
            store,
            "_resolve_payload_handler",
            lambda _manifest: (_ for _ in ()).throw(
                AssertionError("reconciliation must not resolve a payload handler")
            ),
        )
        first = store.reconcile()
        second = store.reconcile()

        assert first.to_dict() == second.to_dict()
        assert store.lifecycle_authority.snapshot_state() == before
        machine_view = first.machine_view()
        assert machine_view["schema_version"] == 2
        assert machine_view["findings"]
        assert set(machine_view["findings"][0]) == {
            "applied_state",
            "authoritative_generation",
            "authority_revision",
            "checkpoint_state",
            "disposition",
            "expected_generation",
            "finding_id",
            "key_fingerprint",
            "operation_provenance",
            "proposed_action",
            "reason_code",
            "residue_role",
            "residue_type",
            "run_revision",
        }
        assert "report-key" not in repr(machine_view)
        with pytest.raises(ValueError, match="Unsupported reconciliation report version"):
            first.machine_view(version=99)
    finally:
        store.close()


def test_authority_reconciliation_apply_resumes_bounded_cleanup_debt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signed resume token continues each captured debt row exactly once."""
    config = CacheConfig(
        lifecycle_limits=LifecycleLimits(
            operation_page_size=1,
            max_reconcile_actions=1,
        )
    )
    store = BlobStore(tmp_path / "resumable-authority-report", backend="json", config=config)
    try:
        store.put({"generation": "old-a"}, key="a")
        store.put({"generation": "old-b"}, key="b")
        original_delete = store._delete_or_prove_absent
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            lambda _locator: (_ for _ in ()).throw(OSError("defer cleanup")),
        )
        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.put({"generation": "new-a"}, key="a")
        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.put({"generation": "new-b"}, key="b")

        monkeypatch.setattr(store, "_delete_or_prove_absent", original_delete)
        first = store.reconcile(apply=True)
        assert first.resume_token is not None
        assert len(store.lifecycle_authority.pending_cleanup_debts()) == 1

        second = store.reconcile(apply=True, resume_token=first.resume_token)
        assert second.resume_token is None
        assert store.lifecycle_authority.pending_cleanup_debts() == ()
        assert store.get("a") == {"generation": "new-a"}
        assert store.get("b") == {"generation": "new-b"}
    finally:
        store.close()
